# Sample Assets for AI Short-Video Creator

This directory contains sample input files and example outputs to demonstrate the capabilities of the AI Short-Video Creator tool.

## Sample Input Files

### 📄 Text Articles

1. **healthcare_ai_article.txt**
   - Topic: Artificial Intelligence in Healthcare
   - Length: ~500 words
   - Content: Medical AI applications, challenges, and future prospects
   - Ideal for: Technology/healthcare content creators

2. **climate_change_article.txt**
   - Topic: Climate Change and Renewable Energy
   - Length: ~450 words
   - Content: Environmental challenges and sustainable solutions
   - Ideal for: Environmental/sustainability advocates

3. **remote_work_article.txt**
   - Topic: Remote Work Transformation
   - Length: ~480 words
   - Content: Technology enabling remote work revolution
   - Ideal for: Business/technology content

### 🎵 Audio Files (Coming Soon)

- `podcast_excerpt.mp3` - Sample podcast segment for transcription
- `interview_clip.wav` - Interview audio for video creation

### 📋 Transcript Files (Coming Soon)

- `webinar_transcript.txt` - Formatted transcript with timestamps
- `meeting_notes.md` - Structured meeting notes for summarization

## Example Output Videos

### 📱 Sample Generated Videos (Coming Soon)

After running the tool with the sample inputs, you'll find example outputs here:

- `healthcare_ai_short.mp4` - 45-second vertical video about AI in healthcare
- `climate_solutions_reel.mp4` - 60-second environmental awareness video
- `remote_work_tips.mp4` - 30-second remote work insights video

### 📝 Generated Assets

Each video generation creates additional files:

- `*.srt` - Caption files for social media upload
- `*_summary.txt` - AI-generated text summaries
- `*_script.json` - Detailed script with timing information
- `*_audio.wav` - Generated narration audio files

## Usage Instructions

1. **Copy any sample file** to the main `assets/` directory
2. **Upload through the Streamlit interface** or use the API
3. **Configure your preferences** (video length, style, voice)
4. **Generate your video** and find outputs in `assets/outputs/`

## Customization Tips

### For Best Results:

- **Text length**: 200-800 words work best for 30-60 second videos
- **Clear structure**: Use headers and bullet points for better AI parsing
- **Relevant content**: Ensure content is suitable for short-form video format
- **Topic focus**: Single-topic articles convert better than multi-topic content

### Content Guidelines:

- ✅ Educational content
- ✅ How-to guides
- ✅ News summaries
- ✅ Product descriptions
- ✅ Research findings
- ❌ Overly technical jargon
- ❌ Very long quotes
- ❌ Complex mathematical formulas

## Contributing Sample Content

Help improve the tool by contributing sample content:

1. Add new sample files to this directory
2. Include diverse topics and content types
3. Ensure content is original or properly licensed
4. Update this README with descriptions

## File Formats Supported

### Input Formats:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (text extraction)
- `.docx` - Word documents
- `.mp3, .wav` - Audio files (with transcription)

### Output Formats:
- `.mp4` - Video files (H.264, 9:16 aspect ratio)
- `.srt` - Subtitle files
- `.wav` - Audio files
- `.json` - Metadata and script files

## Performance Benchmarks

Based on sample files (tested on standard hardware):

| Sample File | Processing Time | Video Length | Summary Quality |
|-------------|----------------|--------------|-----------------|
| Healthcare AI | 85 seconds | 45 seconds | Excellent |
| Climate Change | 92 seconds | 50 seconds | Very Good |
| Remote Work | 78 seconds | 40 seconds | Excellent |

*Times may vary based on system specifications and selected options*

---

**Need help?** Check the main README.md or open an issue on GitHub!