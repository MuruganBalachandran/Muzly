# Muzly Music Recommender System

## Overview
This directory contains the core model components of the Muzly Music Recommender system, which uses computer vision and machine learning to recommend music based on emotions, context, and environmental factors.

## UI Images
<img width="1904" height="893" alt="Image" src="https://github.com/user-attachments/assets/928d88a4-bdd7-4761-8694-57057ebc9b54" />
<img width="1911" height="780" alt="Image" src="https://github.com/user-attachments/assets/0e1f2e9a-4cde-49e0-81c5-befff41075f5" />
<img width="1907" height="596" alt="Image" src="https://github.com/user-attachments/assets/7a8d1373-dba1-4430-85cb-5f531b21df57" />
<img width="1914" height="643" alt="Image" src="https://github.com/user-attachments/assets/a3b84084-0fb7-4f86-978c-eeb1df1dfd08" />
<img width="1901" height="913" alt="Image" src="https://github.com/user-attachments/assets/6c6b8fc7-c581-4f6a-b4b7-7954bd0edf1e" />
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
