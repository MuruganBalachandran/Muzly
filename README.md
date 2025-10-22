# Muzly Music Recommender Model

## Overview
This directory contains the core model components of the Muzly Music Recommender system, which uses computer vision and machine learning to recommend music based on emotions, context, and environmental factors.

## Key Components

### 1. Emotion Detection Model
- **File**: `best_emotion_model.h5`
- **Type**: CNN-based deep learning model
- **Purpose**: Detects 7 emotions from facial expressions
  - Happy
  - Sad
  - Angry
  - Disgust
  - Fear
  - Surprise
  - Neutral

### 2. Main Application
- **File**: `app.py`
- **Features**:
  - Emotion detection from images
  - Scene context analysis
  - Weather and climate detection
  - Intelligent music recommendation system
  - Multi-language support with fallback options
  - Location-aware recommendations

### 3. Dataset
- **File**: `songs.csv`
- **Contents**: Curated music database with:
  - Song names and artists
  - Emotional categorization
  - Context labels
  - Language information
  - Climate/weather associations

## Model Architecture

### Emotion Detection
- Uses a pre-trained CNN architecture
- Input: 48x48 grayscale face images
- Output: 7-class emotion probability distribution

### Scene Analysis
- CLIP-based image understanding
- Context classification for better music matching
- Climate and environmental factor detection

## Setup Instructions

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Files**
   - Ensure `best_emotion_model.h5` is present
   - Check `emotion_model.h5` for backup/alternative model

3. **Running the Application**
   ```bash
   streamlit run app.py
   ```

## Dependencies
- TensorFlow 2.x
- OpenCV
- Streamlit
- Pandas
- PIL
- CLIP
- NumPy

## Usage Notes
1. The system requires clear facial images for accurate emotion detection
2. Good lighting conditions improve accuracy
3. Multiple faces in an image may affect results
4. Internet connection needed for weather API and location services

## Performance Optimization
- Model uses cached resources for faster loading
- Efficient image processing pipeline
- Optimized recommendation algorithm with fallbacks

## Error Handling
- Graceful fallbacks for missing songs
- Alternative recommendations when exact matches unavailable
- Clear user feedback for improving results

## Contributing
1. Follow PEP 8 style guidelines
2. Document any model changes
3. Test thoroughly before committing
4. Update this README as needed

## Maintenance
- Regularly update the songs database
- Monitor model performance
- Keep dependencies up to date
- Check for API limits and quotas