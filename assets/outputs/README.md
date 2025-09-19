# Output Directory

This directory will contain generated videos and associated files created by the AI Short-Video Creator.

## Generated Files Structure

When you create a video, the following files will be generated:

```
outputs/
├── video_YYYYMMDD_HHMMSS/
│   ├── final_video.mp4          # Main output video (9:16 format)
│   ├── captions.srt             # Subtitle file for social media
│   ├── audio_narration.wav      # Generated TTS audio
│   ├── summary.txt              # AI-generated text summary
│   ├── script.json              # Detailed script with timing
│   ├── metadata.json            # Video metadata and settings
│   └── frames/                  # Individual video frames (optional)
│       ├── frame_001.png
│       ├── frame_002.png
│       └── ...
```

## File Descriptions

### 📹 **final_video.mp4**
- Format: MP4 (H.264)
- Resolution: 1080x1920 (9:16 aspect ratio)
- Duration: 30-60 seconds
- Audio: 44.1kHz stereo
- Ready for social media upload

### 📝 **captions.srt**
- Standard SubRip format
- Perfect timing synchronization
- Compatible with all major platforms
- Use for manual subtitle upload

### 🎵 **audio_narration.wav**
- High-quality TTS audio
- Can be used separately for podcasts
- Multiple voice options available

### 📄 **summary.txt**
- AI-extracted key points
- Optimized for video format
- Word count suitable for narration

### 📊 **script.json**
- Detailed timing information
- Scene descriptions
- Audio cue points
- Technical metadata

### ⚙️ **metadata.json**
- Generation settings used
- Model versions
- Processing time
- Quality metrics

## Automatic Cleanup

- Files older than 30 days are automatically archived
- Set `AUTO_CLEANUP=false` in config to disable
- Manual cleanup: `python scripts/cleanup.py`

## Sharing and Export

Generated videos are optimized for:
- ✅ Instagram Reels
- ✅ YouTube Shorts  
- ✅ TikTok
- ✅ LinkedIn Video
- ✅ Twitter Video
- ✅ Facebook Stories

## Troubleshooting

If output files are missing:
1. Check processing logs in console
2. Verify input file format compatibility
3. Ensure sufficient disk space (min 500MB)
4. Check file permissions

---

**Ready to create your first video?** Go back to the main app and upload your content!