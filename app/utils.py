"""
Utility functions for the AI Short-Video Creator.
Handles text processing, file operations, validation, and format conversions.
"""

import os
import re
import math
import hashlib
import tempfile
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from datetime import datetime

import numpy as np
from PIL import Image
import cv2


logger = logging.getLogger(__name__)


def validate_text_input(text: str) -> Dict[str, Union[bool, str, int, float]]:
    """
    Validate and analyze input text for video generation.
    
    Args:
        text: Input text to validate
        
    Returns:
        Dictionary containing validation results and text statistics
    """
    result = {
        'valid': False,
        'message': '',
        'char_count': 0,
        'word_count': 0,
        'sentence_count': 0,
        'paragraph_count': 0,
        'reading_time_min': 0.0,
        'complexity_score': 0.0
    }
    
    if not text or not text.strip():
        result['message'] = "Text is empty"
        return result
    
    text = text.strip()
    
    # Basic counts
    result['char_count'] = len(text)
    result['word_count'] = len(text.split())
    result['sentence_count'] = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
    result['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
    
    # Reading time (average 200 WPM)
    result['reading_time_min'] = result['word_count'] / 200.0
    
    # Complexity score (based on sentence length and word complexity)
    avg_sentence_length = result['word_count'] / max(result['sentence_count'], 1)
    long_words = len([w for w in text.split() if len(w) > 6])
    result['complexity_score'] = (avg_sentence_length + long_words / result['word_count']) / 2
    
    # Validation checks
    if result['char_count'] < 50:
        result['message'] = "Text too short (minimum 50 characters recommended)"
    elif result['char_count'] > 50000:
        result['message'] = "Text too long (maximum 50,000 characters recommended)"
    elif result['word_count'] < 20:
        result['message'] = "Text too short (minimum 20 words recommended)"
    elif result['word_count'] > 10000:
        result['message'] = "Text too long (maximum 10,000 words recommended)"
    else:
        result['valid'] = True
        result['message'] = "Text validation passed"
    
    return result


def estimate_processing_time(
    text_length: int, 
    num_images: int, 
    use_stable_diffusion: bool = True,
    use_gpu: bool = None
) -> Dict[str, float]:
    """
    Estimate processing time for video generation based on input parameters.
    
    Args:
        text_length: Length of input text in characters
        num_images: Number of images to generate
        use_stable_diffusion: Whether to use Stable Diffusion for images
        use_gpu: Whether GPU is available (auto-detect if None)
        
    Returns:
        Dictionary with time estimates for each processing step
    """
    if use_gpu is None:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    
    # Base times in minutes (GPU vs CPU)
    if use_gpu:
        base_times = {
            'summarization_per_1k_chars': 0.1,
            'tts_per_100_words': 0.5,
            'sd_image_generation': 0.5,
            'stock_image_fetch': 0.1,
            'video_assembly_base': 1.0,
            'video_assembly_per_image': 0.2
        }
    else:
        base_times = {
            'summarization_per_1k_chars': 0.3,
            'tts_per_100_words': 1.0,
            'sd_image_generation': 3.0,
            'stock_image_fetch': 0.1,
            'video_assembly_base': 2.0,
            'video_assembly_per_image': 0.3
        }
    
    # Calculate estimates
    word_count = text_length / 5  # Approximate words from character count
    
    summarization_time = (text_length / 1000) * base_times['summarization_per_1k_chars']
    tts_time = (word_count / 100) * base_times['tts_per_100_words']
    
    if use_stable_diffusion and use_gpu:
        image_time = num_images * base_times['sd_image_generation']
    else:
        image_time = num_images * base_times['stock_image_fetch']
    
    video_time = base_times['video_assembly_base'] + (num_images * base_times['video_assembly_per_image'])
    
    total_time = summarization_time + tts_time + image_time + video_time
    
    return {
        'summarization': summarization_time,
        'tts': tts_time,
        'images': image_time,
        'video': video_time,
        'total_minutes': total_time,
        'total_seconds': total_time * 60,
        'use_gpu': use_gpu
    }


def clean_filename(filename: str, max_length: int = 50) -> str:
    """
    Clean and sanitize filename for safe file system usage.
    
    Args:
        filename: Raw filename
        max_length: Maximum length for the filename
        
    Returns:
        Cleaned filename safe for file system
    """
    # Remove or replace unsafe characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple spaces and replace with single underscore
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Remove leading/trailing dots and spaces
    cleaned = cleaned.strip('. ')
    
    # Limit length
    if len(cleaned) > max_length:
        # Try to preserve file extension
        parts = cleaned.rsplit('.', 1)
        if len(parts) == 2 and len(parts[1]) <= 10:
            # Has extension
            base_length = max_length - len(parts[1]) - 1
            cleaned = parts[0][:base_length] + '.' + parts[1]
        else:
            cleaned = cleaned[:max_length]
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "untitled"
    
    return cleaned


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    
    if i >= len(size_names):
        i = len(size_names) - 1
    
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def calculate_text_hash(text: str) -> str:
    """
    Calculate SHA-256 hash of text for caching and deduplication.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def split_text_into_chunks(
    text: str, 
    max_chunk_size: int = 1000, 
    overlap: int = 100,
    split_on_sentences: bool = True
) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        split_on_sentences: Whether to try splitting on sentence boundaries
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    if split_on_sentences:
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        # Simple character-based splitting
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good breaking point (space or punctuation)
            break_point = end
            for i in range(end, max(start + max_chunk_size - 200, start), -1):
                if text[i] in ' \n\t.!?':
                    break_point = i
                    break
            
            chunks.append(text[start:break_point].strip())
            start = max(break_point - overlap, start + 1)
    
    return [chunk for chunk in chunks if chunk.strip()]


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for image prompt generation.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Simple keyword extraction based on word frequency and length
    
    # Clean text
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned_text.split()
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 
        'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself',
        'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
    }
    
    # Filter words
    filtered_words = [
        word for word in words 
        if len(word) > 3 and word not in stop_words and word.isalpha()
    ]
    
    # Count frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score words (frequency * length)
    word_scores = {
        word: freq * len(word) 
        for word, freq in word_freq.items()
    }
    
    # Get top keywords
    keywords = sorted(word_scores.keys(), key=lambda x: word_scores[x], reverse=True)
    
    return keywords[:max_keywords]


def generate_image_prompts_from_text(text: str, num_prompts: int = 3) -> List[str]:
    """
    Generate image prompts from text content.
    
    Args:
        text: Input text content
        num_prompts: Number of prompts to generate
        
    Returns:
        List of image generation prompts
    """
    keywords = extract_keywords(text, max_keywords=15)
    
    # Prompt templates for different types of content
    templates = [
        "professional illustration of {concept}, modern clean style, high quality",
        "abstract representation of {concept}, minimalist design, corporate aesthetic",
        "infographic style visualization of {concept}, clear and informative",
        "conceptual art about {concept}, professional presentation style",
        "modern digital artwork representing {concept}, sleek design",
        "business presentation slide about {concept}, professional layout"
    ]
    
    prompts = []
    
    # Generate prompts using keywords
    for i in range(num_prompts):
        if i < len(keywords):
            concept = keywords[i]
        else:
            # Combine multiple keywords
            concept = " and ".join(keywords[:min(3, len(keywords))])
        
        template = templates[i % len(templates)]
        prompt = template.format(concept=concept)
        prompts.append(prompt)
    
    return prompts


def resize_image_to_aspect_ratio(
    image_path: str, 
    target_width: int = 608, 
    target_height: int = 1080,
    output_path: Optional[str] = None
) -> str:
    """
    Resize image to specific aspect ratio (9:16 for social media).
    
    Args:
        image_path: Path to input image
        target_width: Target width in pixels
        target_height: Target height in pixels
        output_path: Output path (overwrites input if None)
        
    Returns:
        Path to resized image
    """
    if output_path is None:
        output_path = image_path
    
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling factor
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
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to exact target size from center
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            # Save
            img_cropped.save(output_path, quality=95, optimize=True)
            
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to resize image {image_path}: {e}")
        raise


def create_text_overlay_image(
    text: str,
    width: int = 608,
    height: int = 1080,
    background_color: str = "#1a1a1a",
    text_color: str = "#ffffff",
    font_size: int = 40,
    output_path: Optional[str] = None
) -> str:
    """
    Create an image with text overlay for video slides.
    
    Args:
        text: Text to display
        width: Image width
        height: Image height
        background_color: Background color (hex)
        text_color: Text color (hex)
        font_size: Font size
        output_path: Output file path
        
    Returns:
        Path to created image
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.png')
    
    try:
        # Create image
        img = Image.new('RGB', (width, height), color=background_color)
        
        # Try to use a better font
        try:
            from PIL import ImageFont, ImageDraw
            
            # Try different font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/Windows/Fonts/arial.ttf",
                "arial.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except (IOError, OSError):
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except ImportError:
            font = None
        
        draw = ImageDraw.Draw(img)
        
        # Word wrap text
        words = text.split()
        lines = []
        current_line = []
        max_width = width - 80  # Padding
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font:
                text_width = draw.textbbox((0, 0), test_line, font=font)[2]
            else:
                text_width = len(test_line) * (font_size // 2)  # Rough estimate
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate total text height
        line_height = font_size + 10
        total_height = len(lines) * line_height
        
        # Center text vertically
        start_y = (height - total_height) // 2
        
        # Draw text
        for i, line in enumerate(lines):
            if font:
                text_width = draw.textbbox((0, 0), line, font=font)[2]
            else:
                text_width = len(line) * (font_size // 2)
            
            x = (width - text_width) // 2
            y = start_y + (i * line_height)
            
            if font:
                draw.text((x, y), line, fill=text_color, font=font)
            else:
                draw.text((x, y), line, fill=text_color)
        
        # Save image
        img.save(output_path, quality=95, optimize=True)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create text overlay image: {e}")
        raise


def validate_video_output(video_path: str) -> Dict[str, Union[bool, str, float]]:
    """
    Validate the generated video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Validation results
    """
    result = {
        'valid': False,
        'file_exists': False,
        'file_size': 0,
        'duration': 0.0,
        'width': 0,
        'height': 0,
        'fps': 0.0,
        'message': ''
    }
    
    try:
        # Check file existence
        if not os.path.exists(video_path):
            result['message'] = "Video file does not exist"
            return result
        
        result['file_exists'] = True
        result['file_size'] = os.path.getsize(video_path)
        
        # Use OpenCV to get video properties
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            result['message'] = "Cannot open video file"
            return result
        
        # Get video properties
        result['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result['fps'] = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if result['fps'] > 0:
            result['duration'] = frame_count / result['fps']
        
        cap.release()
        
        # Validation checks
        if result['file_size'] == 0:
            result['message'] = "Video file is empty"
        elif result['duration'] == 0:
            result['message'] = "Video has no duration"
        elif result['width'] == 0 or result['height'] == 0:
            result['message'] = "Invalid video dimensions"
        else:
            result['valid'] = True
            result['message'] = "Video validation passed"
        
    except Exception as e:
        result['message'] = f"Video validation error: {str(e)}"
    
    return result


def create_project_metadata(
    project_name: str,
    input_text: str,
    config: Dict,
    results: Dict,
    output_dir: str
) -> str:
    """
    Create comprehensive project metadata file.
    
    Args:
        project_name: Name of the project
        input_text: Original input text
        config: Generation configuration
        results: Generation results
        output_dir: Output directory
        
    Returns:
        Path to metadata file
    """
    metadata = {
        'project_info': {
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'generator': 'AI Short-Video Creator'
        },
        'input': {
            'text_length': len(input_text),
            'text_hash': calculate_text_hash(input_text),
            'validation': validate_text_input(input_text)
        },
        'configuration': config,
        'processing': {
            'estimated_time': estimate_processing_time(
                len(input_text),
                config.get('num_images', 3),
                config.get('use_stable_diffusion', True)
            )
        },
        'results': results,
        'files': {
            'video': os.path.basename(results.get('video_path', '')),
            'audio': os.path.basename(results.get('audio_path', '')),
            'captions': os.path.basename(results.get('captions_path', '')),
            'images': [os.path.basename(img) for img in results.get('images', [])]
        }
    }
    
    # Add video validation if available
    if 'video_path' in results:
        metadata['validation'] = validate_video_output(results['video_path'])
    
    metadata_path = os.path.join(output_dir, 'project_metadata.json')
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_path
        
    except Exception as e:
        logger.error(f"Failed to create metadata file: {e}")
        raise


def cleanup_temp_files(temp_dir: str, keep_final_outputs: bool = True):
    """
    Clean up temporary files while optionally keeping final outputs.
    
    Args:
        temp_dir: Temporary directory to clean
        keep_final_outputs: Whether to keep final video and related files
    """
    try:
        if not os.path.exists(temp_dir):
            return
        
        if keep_final_outputs:
            # Keep only essential files
            essential_files = {
                'final_video.mp4', 'narration.wav', 'captions.srt', 
                'metadata.json', 'project_metadata.json'
            }
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file not in essential_files and not file.startswith('generated_image_'):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {e}")
        else:
            # Remove entire directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


# Export all utility functions
__all__ = [
    'validate_text_input',
    'estimate_processing_time',
    'clean_filename',
    'format_file_size',
    'calculate_text_hash',
    'split_text_into_chunks',
    'extract_keywords',
    'generate_image_prompts_from_text',
    'resize_image_to_aspect_ratio',
    'create_text_overlay_image',
    'validate_video_output',
    'create_project_metadata',
    'cleanup_temp_files'
]