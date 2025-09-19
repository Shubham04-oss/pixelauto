"""
AI Short-Video Creator Pipeline
Converts articles/transcripts into short social-ready videos with AI-generated content.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    pipeline, BlipProcessor, BlipForConditionalGeneration
)
from diffusers import StableDiffusionPipeline
import whisper
from TTS.api import TTS
import moviepy.editor as mp
from moviepy.video.fx.all import resize
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSummarizer:
    """Handles text summarization using HuggingFace transformers."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the summarization model."""
        try:
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded summarization model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            # Fallback to a smaller model
            self.model_name = "sshleifer/distilbart-cnn-12-6"
            self.summarizer = pipeline("summarization", model=self.model_name)
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize the input text for short video content.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text suitable for short video
        """
        try:
            # Split long text into chunks if necessary
            max_chunk_length = 1024
            if len(text) > max_chunk_length:
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.strip()) > 100:  # Skip very short chunks
                        summary = self.summarizer(
                            chunk,
                            max_length=max_length//len(chunks),
                            min_length=min_length//len(chunks),
                            do_sample=False
                        )[0]['summary_text']
                        summaries.append(summary)
                
                # Combine summaries and summarize again if needed
                combined = " ".join(summaries)
                if len(combined) > max_length * 2:
                    final_summary = self.summarizer(
                        combined,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    return final_summary
                else:
                    return combined
            else:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                return summary
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple truncation
            return text[:max_length * 4] + "..."


class TextToSpeech:
    """Handles text-to-speech conversion using Coqui TTS."""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None
        self._load_model()
    
    def _load_model(self):
        """Load the TTS model."""
        try:
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            logger.info(f"Loaded TTS model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            # Fallback to a simpler model or pyttsx3
            try:
                self.tts = TTS("tts_models/en/ljspeech/glow-tts")
            except:
                logger.warning("Using fallback TTS engine")
                import pyttsx3
                self.tts = pyttsx3.init()
    
    def generate_speech(self, text: str, output_path: str) -> str:
        """
        Generate speech audio from text.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            
        Returns:
            Path to the generated audio file
        """
        try:
            if hasattr(self.tts, 'tts_to_file'):
                # Coqui TTS
                self.tts.tts_to_file(text=text, file_path=output_path)
            else:
                # Fallback pyttsx3
                self.tts.save_to_file(text, output_path)
                self.tts.runAndWait()
            
            logger.info(f"Generated speech audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None


class ImageGenerator:
    """Handles image generation for video visuals."""
    
    def __init__(self, use_stable_diffusion: bool = True):
        self.use_stable_diffusion = use_stable_diffusion
        self.sd_pipeline = None
        self.pexels_api_key = os.getenv('PEXELS_API_KEY')
        
        if use_stable_diffusion and torch.cuda.is_available():
            self._load_stable_diffusion()
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion pipeline for image generation."""
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.sd_pipeline.to("cuda")
            logger.info("Loaded Stable Diffusion pipeline")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")
            self.sd_pipeline = None
    
    def generate_images(self, prompts: List[str], output_dir: str) -> List[str]:
        """
        Generate images based on text prompts.
        
        Args:
            prompts: List of text prompts for image generation
            output_dir: Directory to save generated images
            
        Returns:
            List of paths to generated images
        """
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            output_path = os.path.join(output_dir, f"generated_image_{i}.png")
            
            if self.sd_pipeline:
                try:
                    # Generate with Stable Diffusion
                    image = self.sd_pipeline(
                        prompt,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        height=1080,
                        width=608  # 9:16 aspect ratio
                    ).images[0]
                    
                    image.save(output_path)
                    image_paths.append(output_path)
                    logger.info(f"Generated image with SD: {output_path}")
                    
                except Exception as e:
                    logger.error(f"SD generation failed for prompt '{prompt}': {e}")
                    # Fallback to stock photos
                    stock_path = self._get_stock_image(prompt, output_path)
                    if stock_path:
                        image_paths.append(stock_path)
            else:
                # Use stock photos or create simple visuals
                stock_path = self._get_stock_image(prompt, output_path)
                if stock_path:
                    image_paths.append(stock_path)
                else:
                    # Create a simple colored background with text
                    self._create_text_slide(prompt, output_path)
                    image_paths.append(output_path)
        
        return image_paths
    
    def _get_stock_image(self, query: str, output_path: str) -> Optional[str]:
        """Fetch stock image from Pexels API."""
        if not self.pexels_api_key:
            return None
            
        try:
            headers = {'Authorization': self.pexels_api_key}
            url = f"https://api.pexels.com/v1/search?query={query}&per_page=1&orientation=portrait"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['photos']:
                    photo_url = data['photos'][0]['src']['large2x']
                    
                    # Download image
                    img_response = requests.get(photo_url)
                    if img_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(img_response.content)
                        
                        # Resize to 9:16 aspect ratio
                        self._resize_to_portrait(output_path)
                        return output_path
                        
        except Exception as e:
            logger.error(f"Failed to fetch stock image for '{query}': {e}")
        
        return None
    
    def _create_text_slide(self, text: str, output_path: str):
        """Create a simple text slide as fallback visual."""
        # Create a 9:16 aspect ratio image
        width, height = 608, 1080
        image = Image.new('RGB', (width, height), color='#1a1a1a')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Wrap text
        words = text.split()
        lines = []
        current_line = []
        max_width = width - 40
        
        for word in words:
            current_line.append(word)
            line_text = ' '.join(current_line)
            if draw.textsize(line_text, font=font)[0] > max_width:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text
        total_height = len(lines) * 50
        y = (height - total_height) // 2
        
        for line in lines:
            text_width = draw.textsize(line, font=font)[0]
            x = (width - text_width) // 2
            draw.text((x, y), line, fill='white', font=font)
            y += 50
        
        image.save(output_path)
    
    def _resize_to_portrait(self, image_path: str):
        """Resize image to 9:16 portrait aspect ratio."""
        try:
            with Image.open(image_path) as img:
                # Target size for 9:16 aspect ratio
                target_width, target_height = 608, 1080
                
                # Calculate scaling to fit within target while maintaining aspect ratio
                img_ratio = img.width / img.height
                target_ratio = target_width / target_height
                
                if img_ratio > target_ratio:
                    # Image is too wide, scale by height
                    new_height = target_height
                    new_width = int(target_height * img_ratio)
                else:
                    # Image is too tall, scale by width
                    new_width = target_width
                    new_height = int(target_width / img_ratio)
                
                # Resize and crop to center
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Crop to exact target size
                left = (new_width - target_width) // 2
                top = (new_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                
                img_cropped = img_resized.crop((left, top, right, bottom))
                img_cropped.save(image_path)
                
        except Exception as e:
            logger.error(f"Failed to resize image {image_path}: {e}")


class CaptionGenerator:
    """Handles caption generation and timing alignment."""
    
    def __init__(self):
        self.whisper_model = None
    
    def _load_whisper(self):
        """Load Whisper model for forced alignment."""
        if not self.whisper_model:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Loaded Whisper model for alignment")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
    
    def generate_captions(self, text: str, audio_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Generate SRT captions and timing alignment.
        
        Args:
            text: The text that was converted to speech
            audio_path: Path to the audio file
            output_dir: Directory to save caption files
            
        Returns:
            Tuple of (srt_path, timing_data_path)
        """
        srt_path = os.path.join(output_dir, "captions.srt")
        timing_path = os.path.join(output_dir, "timing.json")
        
        try:
            # Load Whisper for forced alignment
            self._load_whisper()
            
            if self.whisper_model:
                # Use Whisper for better alignment
                result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
                segments = result.get('segments', [])
                
                # Generate SRT from segments
                self._write_srt_from_segments(segments, srt_path)
                
                # Save timing data
                timing_data = {
                    'segments': segments,
                    'total_duration': result.get('duration', 0)
                }
                with open(timing_path, 'w') as f:
                    json.dump(timing_data, f, indent=2)
                    
            else:
                # Fallback: estimate timing based on text length and audio duration
                self._estimate_timing(text, audio_path, srt_path, timing_path)
            
            logger.info(f"Generated captions: {srt_path}")
            return srt_path, timing_path
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            # Create basic captions
            self._create_basic_captions(text, srt_path, timing_path)
            return srt_path, timing_path
    
    def _write_srt_from_segments(self, segments: List[Dict], srt_path: str):
        """Write SRT file from Whisper segments."""
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._seconds_to_srt_time(segment['start'])
                end_time = self._seconds_to_srt_time(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _estimate_timing(self, text: str, audio_path: str, srt_path: str, timing_path: str):
        """Estimate timing based on text length and audio duration."""
        try:
            # Get audio duration
            audio = mp.AudioFileClip(audio_path)
            duration = audio.duration
            audio.close()
            
            # Split text into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text]
            
            # Estimate timing per sentence
            time_per_sentence = duration / len(sentences)
            
            # Generate SRT
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, sentence in enumerate(sentences):
                    start_time = i * time_per_sentence
                    end_time = (i + 1) * time_per_sentence
                    
                    start_srt = self._seconds_to_srt_time(start_time)
                    end_srt = self._seconds_to_srt_time(end_time)
                    
                    f.write(f"{i + 1}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{sentence}\n\n")
            
            # Save timing data
            timing_data = {
                'estimated': True,
                'total_duration': duration,
                'sentences': len(sentences),
                'time_per_sentence': time_per_sentence
            }
            with open(timing_path, 'w') as f:
                json.dump(timing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Timing estimation failed: {e}")
            self._create_basic_captions(text, srt_path, timing_path)
    
    def _create_basic_captions(self, text: str, srt_path: str, timing_path: str):
        """Create basic captions without timing."""
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("1\n")
            f.write("00:00:00,000 --> 00:01:00,000\n")
            f.write(f"{text}\n\n")
        
        timing_data = {'basic': True, 'text': text}
        with open(timing_path, 'w') as f:
            json.dump(timing_data, f, indent=2)


class VideoCreator:
    """Handles final video composition and export."""
    
    def __init__(self):
        self.target_width = 608  # 9:16 aspect ratio
        self.target_height = 1080
    
    def create_video(
        self, 
        audio_path: str, 
        images: List[str], 
        captions_path: str,
        output_path: str,
        background_color: str = "#000000"
    ) -> str:
        """
        Create the final video by combining audio, images, and captions.
        
        Args:
            audio_path: Path to the audio file
            images: List of image paths for visuals
            captions_path: Path to SRT captions file
            output_path: Path for the output video
            background_color: Background color for the video
            
        Returns:
            Path to the created video
        """
        try:
            logger.info("Starting video creation...")
            
            # Load audio to get duration
            audio_clip = mp.AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # Create video clips from images
            video_clips = []
            if images:
                clip_duration = duration / len(images)
                
                for i, image_path in enumerate(images):
                    if os.path.exists(image_path):
                        # Load and resize image
                        img_clip = mp.ImageClip(image_path, duration=clip_duration)
                        img_clip = img_clip.resize((self.target_width, self.target_height))
                        img_clip = img_clip.set_start(i * clip_duration)
                        video_clips.append(img_clip)
            
            if not video_clips:
                # Create a simple colored background if no images
                color_clip = mp.ColorClip(
                    size=(self.target_width, self.target_height),
                    color=background_color,
                    duration=duration
                )
                video_clips = [color_clip]
            
            # Combine video clips
            final_video = mp.CompositeVideoClip(video_clips, size=(self.target_width, self.target_height))
            
            # Add audio
            final_video = final_video.set_audio(audio_clip)
            
            # Add captions if available
            if os.path.exists(captions_path):
                try:
                    final_video = self._add_burned_captions(final_video, captions_path)
                except Exception as e:
                    logger.warning(f"Failed to add burned captions: {e}")
            
            # Export video
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Cleanup
            audio_clip.close()
            final_video.close()
            for clip in video_clips:
                clip.close()
            
            logger.info(f"Video created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise
    
    def _add_burned_captions(self, video_clip, captions_path: str):
        """Add burned-in captions to the video."""
        try:
            # Parse SRT file
            captions = self._parse_srt(captions_path)
            
            # Create text clips for captions
            caption_clips = []
            for caption in captions:
                text_clip = mp.TextClip(
                    caption['text'],
                    fontsize=40,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial-Bold'
                ).set_position(('center', 'bottom')).set_start(caption['start']).set_end(caption['end'])
                
                caption_clips.append(text_clip)
            
            # Composite with captions
            if caption_clips:
                video_with_captions = mp.CompositeVideoClip([video_clip] + caption_clips)
                return video_with_captions
            
        except Exception as e:
            logger.error(f"Failed to add captions: {e}")
        
        return video_clip
    
    def _parse_srt(self, srt_path: str) -> List[Dict]:
        """Parse SRT file and return caption data."""
        captions = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                blocks = content.split('\n\n')
                
                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) >= 3:
                        # Parse timing
                        timing_line = lines[1]
                        start_str, end_str = timing_line.split(' --> ')
                        
                        start_time = self._srt_time_to_seconds(start_str)
                        end_time = self._srt_time_to_seconds(end_str)
                        
                        # Get text (join remaining lines)
                        text = ' '.join(lines[2:])
                        
                        captions.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
                        
        except Exception as e:
            logger.error(f"Failed to parse SRT: {e}")
        
        return captions
    
    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format to seconds."""
        try:
            time_part, ms_part = time_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            
            total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
            return total_seconds
            
        except Exception:
            return 0.0


class VideoGenerationPipeline:
    """Main pipeline that orchestrates the entire video generation process."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the video generation pipeline.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        
        # Initialize components
        self.summarizer = TextSummarizer(
            model_name=self.config.get('summarizer_model', 'facebook/bart-large-cnn')
        )
        self.tts = TextToSpeech(
            model_name=self.config.get('tts_model', 'tts_models/en/ljspeech/tacotron2-DDC')
        )
        self.image_generator = ImageGenerator(
            use_stable_diffusion=self.config.get('use_stable_diffusion', True)
        )
        self.caption_generator = CaptionGenerator()
        self.video_creator = VideoCreator()
        
        logger.info("Video generation pipeline initialized")
    
    def process_text_to_video(
        self, 
        input_text: str, 
        output_dir: str,
        video_length: int = 60,
        num_images: int = 3
    ) -> Dict[str, str]:
        """
        Complete pipeline to convert text to short video.
        
        Args:
            input_text: Input article or transcript text
            output_dir: Directory to save all outputs
            video_length: Target video length in seconds
            num_images: Number of images to generate
            
        Returns:
            Dictionary with paths to generated files
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info("Step 1: Summarizing text...")
            # Summarize text for short video format
            summary = self.summarizer.summarize(
                input_text, 
                max_length=min(150, video_length * 2),  # Adjust based on video length
                min_length=50
            )
            
            logger.info("Step 2: Generating speech...")
            # Generate speech audio
            audio_path = os.path.join(output_dir, "narration.wav")
            self.tts.generate_speech(summary, audio_path)
            
            logger.info("Step 3: Generating visual prompts...")
            # Generate image prompts from summary
            image_prompts = self._generate_image_prompts(summary, num_images)
            
            logger.info("Step 4: Creating images...")
            # Generate images
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            image_paths = self.image_generator.generate_images(image_prompts, images_dir)
            
            logger.info("Step 5: Generating captions...")
            # Generate captions and timing
            captions_dir = os.path.join(output_dir, "captions")
            os.makedirs(captions_dir, exist_ok=True)
            srt_path, timing_path = self.caption_generator.generate_captions(
                summary, audio_path, captions_dir
            )
            
            logger.info("Step 6: Creating final video...")
            # Create final video
            video_path = os.path.join(output_dir, "final_video.mp4")
            self.video_creator.create_video(
                audio_path=audio_path,
                images=image_paths,
                captions_path=srt_path,
                output_path=video_path
            )
            
            # Save summary and metadata
            metadata = {
                'original_text_length': len(input_text),
                'summary_length': len(summary),
                'summary': summary,
                'image_prompts': image_prompts,
                'processing_config': self.config
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            result = {
                'video_path': video_path,
                'audio_path': audio_path,
                'captions_path': srt_path,
                'summary': summary,
                'metadata_path': metadata_path,
                'images': image_paths
            }
            
            logger.info(f"Video generation completed! Output: {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _generate_image_prompts(self, text: str, num_prompts: int) -> List[str]:
        """Generate image prompts from text summary."""
        # Simple prompt generation based on key concepts
        words = text.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        key_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Generate prompts
        prompts = []
        
        if len(key_words) >= num_prompts:
            # Use key words to create themed prompts
            for i in range(num_prompts):
                if i < len(key_words):
                    prompt = f"professional illustration of {key_words[i]}, clean modern style, high quality"
                else:
                    prompt = f"abstract concept art about {' '.join(key_words[:3])}, minimalist design"
                prompts.append(prompt)
        else:
            # Create general prompts
            base_prompt = "professional presentation slide about "
            prompts = [
                base_prompt + "technology and innovation, modern clean design",
                base_prompt + "business growth and success, corporate style",
                base_prompt + "digital transformation, futuristic elements"
            ][:num_prompts]
        
        return prompts


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial Intelligence is revolutionizing the way we work and live. Machine learning algorithms 
    can now process vast amounts of data to identify patterns and make predictions with unprecedented 
    accuracy. From healthcare to finance, AI is transforming industries by automating complex tasks 
    and providing insights that were previously impossible to obtain. As we move forward, the 
    integration of AI into our daily lives will continue to accelerate, bringing both opportunities 
    and challenges that we must carefully navigate.
    """
    
    pipeline = VideoGenerationPipeline()
    
    output_dir = "/tmp/ai_video_output"
    result = pipeline.process_text_to_video(
        input_text=sample_text,
        output_dir=output_dir,
        video_length=45,
        num_images=3
    )
    
    print("Video generation completed!")
    print(f"Video saved to: {result['video_path']}")