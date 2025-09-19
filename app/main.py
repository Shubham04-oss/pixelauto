"""
AI Short-Video Creator - Streamlit Interface
A user-friendly web interface for converting articles/transcripts into short social-ready videos.
"""

import streamlit as st
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Optional
import json
import zipfile

from video_pipeline import VideoGenerationPipeline
from utils import (
    validate_text_input, 
    estimate_processing_time,
    format_file_size,
    clean_filename
)

# Page configuration
st.set_page_config(
    page_title="AI Short-Video Creator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4ECDC4;
    }
    
    .stats-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None


def initialize_pipeline(config: Dict) -> VideoGenerationPipeline:
    """Initialize the video generation pipeline with configuration."""
    if st.session_state.pipeline is None:
        with st.spinner("🔧 Initializing AI models... This may take a few minutes on first run."):
            try:
                st.session_state.pipeline = VideoGenerationPipeline(config)
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize pipeline: {str(e)}")
                st.stop()
    
    return st.session_state.pipeline


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎬 AI Short-Video Creator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h4>Transform Articles into Engaging Short Videos! 🚀</h4>
        <p>Convert any text content into professional short-form videos with AI-generated narration, 
        captions, and visuals. Perfect for social media, education, and content marketing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("🤖 AI Models")
        
        summarizer_model = st.selectbox(
            "Summarization Model",
            ["facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6", "google/pegasus-xsum"],
            help="Choose the model for text summarization"
        )
        
        tts_model = st.selectbox(
            "Text-to-Speech Model",
            ["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/en/ljspeech/glow-tts"],
            help="Choose the TTS model for narration"
        )
        
        use_stable_diffusion = st.checkbox(
            "Use Stable Diffusion for Images",
            value=True,
            help="Generate AI images (requires GPU) or use stock photos"
        )
        
        # Video settings
        st.subheader("🎥 Video Settings")
        
        video_length = st.slider(
            "Target Video Length (seconds)",
            min_value=15,
            max_value=180,
            value=60,
            step=15
        )
        
        num_images = st.slider(
            "Number of Visual Slides",
            min_value=1,
            max_value=8,
            value=3
        )
        
        background_color = st.color_picker(
            "Background Color",
            value="#000000"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            custom_prompts = st.text_area(
                "Custom Image Prompts (one per line)",
                placeholder="professional office scene\nmodern technology illustration\nabstract data visualization",
                help="Override automatic prompt generation"
            )
            
            max_summary_length = st.number_input(
                "Maximum Summary Length",
                min_value=50,
                max_value=300,
                value=150,
                step=25
            )
            
            pexels_api_key = st.text_input(
                "Pexels API Key (Optional)",
                type="password",
                help="For stock photos when Stable Diffusion is not available"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📝 Paste Text", "📄 Upload File", "🎤 Upload Audio for Transcription"],
            horizontal=True
        )
        
        input_text = ""
        
        if input_method == "📝 Paste Text":
            input_text = st.text_area(
                "Enter your article or transcript:",
                height=300,
                placeholder="Paste your article, blog post, or transcript here...\n\nThe AI will automatically summarize this content and create a short video with narration, captions, and relevant visuals.",
                help="Minimum 100 characters recommended for good results"
            )
            
        elif input_method == "📄 Upload File":
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt', 'md', 'rtf'],
                help="Supported formats: TXT, MD, RTF"
            )
            
            if uploaded_file is not None:
                try:
                    input_text = uploaded_file.read().decode('utf-8')
                    st.success(f"✅ File uploaded: {uploaded_file.name} ({format_file_size(len(input_text))})")
                    st.text_area("Content preview:", input_text[:500] + "..." if len(input_text) > 500 else input_text, height=150)
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    
        elif input_method == "🎤 Upload Audio for Transcription":
            uploaded_audio = st.file_uploader(
                "Upload an audio file for transcription",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="The AI will transcribe your audio and then create a video"
            )
            
            if uploaded_audio is not None:
                st.info("🔄 Audio transcription will be processed during video generation")
                # We'll handle transcription in the pipeline
                input_text = f"AUDIO_FILE:{uploaded_audio.name}"
        
        # Text validation and stats
        if input_text and not input_text.startswith("AUDIO_FILE:"):
            validation_result = validate_text_input(input_text)
            
            if validation_result['valid']:
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Text validated successfully!</strong><br>
                    📊 Characters: {validation_result['char_count']} | Words: {validation_result['word_count']} | 
                    Estimated reading time: {validation_result['reading_time_min']:.1f} minutes
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>Text validation issues:</strong><br>
                    {validation_result['message']}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Processing Info")
        
        if input_text:
            estimated_time = estimate_processing_time(
                len(input_text) if not input_text.startswith("AUDIO_FILE:") else 5000,
                num_images,
                use_stable_diffusion
            )
            
            st.markdown(f"""
            <div class="stats-box">
                <h4>⏱️ Estimated Processing Time</h4>
                <h2>{estimated_time['total_minutes']:.1f} minutes</h2>
                <hr>
                <small>
                • Summarization: {estimated_time['summarization']:.1f}min<br>
                • Speech Generation: {estimated_time['tts']:.1f}min<br>
                • Image Creation: {estimated_time['images']:.1f}min<br>
                • Video Assembly: {estimated_time['video']:.1f}min
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            if estimated_time['total_minutes'] > 10:
                st.warning("⚠️ Long processing time expected. Consider reducing the number of images or text length.")
        
        # System requirements
        st.markdown("""
        <div class="feature-box">
            <h4>💡 Tips for Best Results</h4>
            <ul>
            <li>🔤 Use 200-2000 words for optimal results</li>
            <li>📰 News articles and how-to content work great</li>
            <li>🎯 Clear, structured text produces better videos</li>
            <li>🖼️ 3-5 images work best for 60-second videos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate button and processing
    st.markdown("---")
    
    col_gen1, col_gen2, col_gen3 = st.columns([1, 2, 1])
    
    with col_gen2:
        generate_button = st.button(
            "🚀 Generate Video",
            type="primary",
            disabled=not input_text or st.session_state.processing,
            use_container_width=True
        )
    
    # Video generation process
    if generate_button and input_text:
        st.session_state.processing = True
        st.session_state.results = None
        
        # Prepare configuration
        config = {
            'summarizer_model': summarizer_model,
            'tts_model': tts_model,
            'use_stable_diffusion': use_stable_diffusion,
            'max_summary_length': max_summary_length
        }
        
        # Set environment variables
        if pexels_api_key:
            os.environ['PEXELS_API_KEY'] = pexels_api_key
        
        try:
            # Initialize pipeline
            pipeline = initialize_pipeline(config)
            
            # Create temporary output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = tempfile.mkdtemp(prefix=f"video_output_{timestamp}_")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎬 Creating your video..."):
                # Handle audio transcription if needed
                if input_text.startswith("AUDIO_FILE:"):
                    status_text.text("🎤 Transcribing audio...")
                    progress_bar.progress(10)
                    # TODO: Implement audio transcription
                    input_text = "This is a placeholder for transcribed audio content."
                
                status_text.text("📝 Summarizing content...")
                progress_bar.progress(20)
                
                # Process custom prompts if provided
                custom_prompt_list = []
                if custom_prompts.strip():
                    custom_prompt_list = [p.strip() for p in custom_prompts.split('\n') if p.strip()]
                
                # Run the pipeline
                results = pipeline.process_text_to_video(
                    input_text=input_text,
                    output_dir=output_dir,
                    video_length=video_length,
                    num_images=num_images
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Video generation completed!")
                
                st.session_state.results = {
                    'output_dir': output_dir,
                    'results': results,
                    'timestamp': timestamp
                }
                
        except Exception as e:
            st.error(f"❌ Error during video generation: {str(e)}")
            st.exception(e)
        finally:
            st.session_state.processing = False
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.header("🎉 Video Generated Successfully!")
        
        results = st.session_state.results['results']
        output_dir = st.session_state.results['output_dir']
        
        # Video preview
        col_video1, col_video2 = st.columns([2, 1])
        
        with col_video1:
            if os.path.exists(results['video_path']):
                st.video(results['video_path'])
            else:
                st.error("❌ Video file not found")
        
        with col_video2:
            st.subheader("📋 Video Details")
            
            # Load metadata
            try:
                with open(results['metadata_path'], 'r') as f:
                    metadata = json.load(f)
                
                st.write(f"**Original Text:** {metadata['original_text_length']} characters")
                st.write(f"**Summary:** {metadata['summary_length']} characters")
                st.write(f"**Images Generated:** {len(results['images'])}")
                
                with st.expander("📝 Generated Summary"):
                    st.write(metadata['summary'])
                
                with st.expander("🖼️ Image Prompts Used"):
                    for i, prompt in enumerate(metadata['image_prompts'], 1):
                        st.write(f"{i}. {prompt}")
                        
            except Exception as e:
                st.warning(f"Could not load metadata: {e}")
        
        # Download section
        st.subheader("⬇️ Download Files")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        
        with col_dl1:
            if os.path.exists(results['video_path']):
                with open(results['video_path'], 'rb') as f:
                    st.download_button(
                        label="📹 Download Video",
                        data=f.read(),
                        file_name=f"ai_video_{st.session_state.results['timestamp']}.mp4",
                        mime="video/mp4"
                    )
        
        with col_dl2:
            if os.path.exists(results['audio_path']):
                with open(results['audio_path'], 'rb') as f:
                    st.download_button(
                        label="🎵 Download Audio",
                        data=f.read(),
                        file_name=f"narration_{st.session_state.results['timestamp']}.wav",
                        mime="audio/wav"
                    )
        
        with col_dl3:
            if os.path.exists(results['captions_path']):
                with open(results['captions_path'], 'r') as f:
                    st.download_button(
                        label="📄 Download Captions",
                        data=f.read(),
                        file_name=f"captions_{st.session_state.results['timestamp']}.srt",
                        mime="text/plain"
                    )
        
        with col_dl4:
            # Create a zip file with all assets
            zip_path = create_asset_zip(output_dir, st.session_state.results['timestamp'])
            if zip_path and os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📦 Download All",
                        data=f.read(),
                        file_name=f"video_package_{st.session_state.results['timestamp']}.zip",
                        mime="application/zip"
                    )
        
        # Generated images gallery
        if results['images']:
            st.subheader("🖼️ Generated Images")
            
            cols = st.columns(min(len(results['images']), 4))
            for i, image_path in enumerate(results['images']):
                with cols[i % len(cols)]:
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"Image {i+1}", use_column_width=True)
        
        # Cleanup option
        if st.button("🗑️ Clear Results"):
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            st.session_state.results = None
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <h4>🚀 AI Short-Video Creator</h4>
        <p>Powered by open-source AI models • Created with ❤️ using Streamlit</p>
        <p><small>
        🤗 HuggingFace Transformers • 🎤 Coqui TTS • 🎨 Stable Diffusion • 🎬 MoviePy • 📝 OpenAI Whisper
        </small></p>
    </div>
    """, unsafe_allow_html=True)


def create_asset_zip(output_dir: str, timestamp: str) -> Optional[str]:
    """Create a zip file containing all generated assets."""
    try:
        zip_path = os.path.join(output_dir, f"video_package_{timestamp}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if not file.endswith('.zip'):  # Don't include the zip itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
        
        return zip_path
    except Exception as e:
        st.error(f"Failed to create zip file: {e}")
        return None


if __name__ == "__main__":
    main()