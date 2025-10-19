import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import pandas as pd
import os
from tensorflow.keras.models import load_model
from transformers.models.clip import CLIPProcessor, CLIPModel
import torch

# -------------------------------
# Streamlit page config MUST be first
# -------------------------------
st.set_page_config(
    page_title="🎵 Muzly Music Recommender",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        /* Global dark background */
        [data-testid="stAppViewContainer"] {
            background-color: #0b0f0e;
            color: #e6f4ee;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        }

        .header-container {
            padding: 1.5rem;
            text-align: center;
            background: linear-gradient(90deg, #064E3B 0%, #0b0f0e 100%);
            color: #ffffff;
            border-radius: 12px;
            margin: 0.5rem 0 1rem 0;
            box-shadow: 0 8px 30px rgba(3, 105, 83, 0.18);
        }

        /* Song card styles (featured + recommendations) */
        .horizontal-scroll {
            display: flex;
            gap: 1rem;
            overflow-x: auto;
            padding: 1rem 0.5rem;
            margin: 1rem 0 2rem 0;
            scrollbar-width: thin;
            scrollbar-color: #064E3B #0b0f0e;
        }

        /* Featured grid (4 in a row on desktop) */
        .featured-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin: 1rem auto 2rem auto;
            max-width: 1200px;
            width: 100%;
            padding: 0 1rem;
        }

        @media (max-width: 1200px) {
            .featured-grid { grid-template-columns: repeat(3, 1fr); }
        }

        @media (max-width: 900px) {
            .featured-grid { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 600px) {
            .featured-grid { grid-template-columns: repeat(1, 1fr); }
        }

        .horizontal-scroll::-webkit-scrollbar { height: 10px; }
        .horizontal-scroll::-webkit-scrollbar-track { background: transparent; }
        .horizontal-scroll::-webkit-scrollbar-thumb { background: #064E3B; border-radius: 8px; }

        .song-card {
            background: linear-gradient(180deg, #151818 0%, #1f2523 100%);
            border-radius: 12px;
            width: 100%;
            padding: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 6px 18px rgba(2,6,23,0.6);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }

        .song-card:hover { transform: translateY(-6px); box-shadow: 0 12px 30px rgba(2,6,23,0.75); }

        .song-cover {
            width: 100%;
            height: 140px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        .song-info { color: #e6f4ee; padding: 0 4px; }
        .song-title { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
        .song-artist { font-size: 0.9rem; color: #bcdccb; margin-bottom: 6px; }
        .song-genre { font-size: 0.8rem; color: #98b99b; margin-bottom: 4px; }
        .song-mood { font-size: 0.8rem; color: #98b99b; margin-bottom: 8px; }
        .song-extra { font-size: 0.75rem; color: #98b99b; margin-top: 4px; }

        .song-actions { display:flex; gap:8px; margin-top:10px; }
        .btn { 
            flex:1; 
            padding:8px 10px; 
            border-radius:8px; 
            display: flex;
            align-items: center;
            justify-content: center;
            cursor:pointer; 
            font-weight:600;
            font-size: 1.2rem;
            min-width: 40px;
            aspect-ratio: 1;
        }
        .btn-play { background: linear-gradient(90deg,#1DB954,#064E3B); color:#fff; }
        .btn-like { background: rgba(255,255,255,0.06); color:#e6f4ee; border:1px solid rgba(255,255,255,0.04); }
        .btn-favorite { background: rgba(255,255,255,0.06); color:#e6f4ee; border:1px solid rgba(255,255,255,0.04); }
        .btn-preview { background: rgba(255,255,255,0.06); color:#e6f4ee; border:1px solid rgba(255,255,255,0.04); }
        .btn-favorite { background: rgba(255,215,0,0.1); color:#ffd700; border:1px solid rgba(255,215,0,0.2); }
        .btn-preview { background: rgba(0,123,255,0.1); color:#007bff; border:1px solid rgba(0,123,255,0.2); }

        .feature-card { background: transparent; padding: 1.25rem; border-radius: 10px; }

        .upload-section { 
            background: linear-gradient(90deg,#08110f,#10221d);
            padding: 2rem; 
            border-radius: 12px; 
            margin: 2rem 0; 
            text-align: center;
            border: 1px solid rgba(6,78,59,0.18);
            color: #e6f4ee;
        }

        .search-container { max-width: 720px; margin: 1.25rem auto; padding: 0.5rem; }
        .stTextInput input { background-color: #0f1614; color: #e6f4ee; border: 1px solid rgba(6,78,59,0.2); border-radius: 10px; padding: 0.85rem; }

        h1,h2,h3,h4,h5 { color: #e6f4ee !important; }
        p, small { color: #cfe9d7 !important; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header-container">
        <h1>🎵 Muzly Music Recommender</h1>
        <p>Experience the perfect harmony of emotion and music</p>
    </div>
""", unsafe_allow_html=True)

# Load and display song database
SONGS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'songs.csv')
songs_df = pd.read_csv(SONGS_CSV_PATH)

# Display the song database in a table
st.markdown("""
    <h2 style='text-align: center; margin: 0.5rem 0;'>📊 Song Database</h2>
""", unsafe_allow_html=True)

st.dataframe(
    songs_df,
    column_config={
        "songname": "Song Name",
        "artist": "Artist",
        "language": "Language",
        "emotion": "Emotion",
        "context": "Context",
        "image_climate": "Image Climate",
        "local_weather": "Local Weather"
    },
    hide_index=True,
    use_container_width=True
)

# Featured Songs Section with Search
st.markdown("""
    <h2 style='text-align: center; margin: 2rem 0 0.5rem 0;'>🎸 Featured Songs</h2>
""", unsafe_allow_html=True)


# Load songs from CSV
@st.cache_data
def load_songs(csv_path):
    df = pd.read_csv(csv_path)
    return df

SONGS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'songs.csv')
songs_df = load_songs(SONGS_CSV_PATH)

# Region to language mapping
def get_language_from_region(location):
    # Indian states mapping
    indian_states = {
        "Tamil Nadu": "Tamil",
        "Karnataka": "Kannada",
        "Kerala": "Malayalam",
        "Andhra Pradesh": "Telugu",
        "Telangana": "Telugu",
        "Maharashtra": "Marathi",
        "West Bengal": "Bengali",
        "Gujarat": "Gujarati",
        "Punjab": "Punjabi"
    }
    
    # International regions mapping
    international_regions = {
        "South Korea": "Korean",
        "Japan": "Japanese",
        "Spain": "Spanish",
        "France": "French",
        "Germany": "German",
        "Italy": "Italian",
        "China": "Chinese",
        "United States": "English",
        "United Kingdom": "English",
        "Australia": "English",
        "Canada": "English"
    }
    
    region = location.get('region', '')
    country = location.get('country', '')
    
    # First try to match Indian states
    if region in indian_states:
        return indian_states[region]
    
    # Then try international mapping
    if country in international_regions:
        return international_regions[country]
        
    return "English"  # Default to English if no specific mapping

# Search bar
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input("🔍 Search songs by title, artist, or genre", key="song_search")
st.markdown('</div>', unsafe_allow_html=True)


# Filter songs based on search (from DataFrame)
if search_query:
    filtered_songs = songs_df[
        songs_df['songname'].str.lower().str.contains(search_query.lower()) |
        songs_df['artist'].str.lower().str.contains(search_query.lower()) |
        songs_df['language'].str.lower().str.contains(search_query.lower())
    ]
else:
    filtered_songs = songs_df

# Display all songs in a 4-column grid
cols = st.columns(4)
for idx, (_, song) in enumerate(filtered_songs.iterrows()):
    col_idx = idx % 4
    with cols[col_idx]:
        cover = f"https://picsum.photos/seed/{song['songname'].replace(' ','')}/400/400"
        st.markdown(f"""
            <div class="song-card">
                <img class="song-cover" src="{cover}" alt="cover" />
                <div class="song-info">
                    <div class="song-title">{song['songname']}</div>
                    <div class="song-artist">by {song['artist']}</div>
                    <div class="song-genre">Language: {song['language']}</div>
                    <div class="song-mood">Mood: {song['emotion'].title()}</div>
                    <div class="song-actions">
                        <div class="btn btn-play">▶</div>
                        <div class="btn btn-like">♡ </div>
                        <div class="btn btn-favorite">★ </div>
                        <div class="btn btn-preview">▶ </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------
# Load Emotion Model
# -------------------------------
@st.cache_resource
def load_emotion_model():
    try:
        model_path = "d:/Final_year_Project/model/best_emotion_model.h5"
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

emotion_model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -------------------------------
# Load CLIP Model
# -------------------------------
@st.cache_resource
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

clip_model, clip_processor = load_clip_model()
scene_labels = ["indoor", "outdoor", "party", "nature", "street", "night", "office",
                "home", "restaurant", "beach", "concert", "mountain", "park"]

# -------------------------------
# Utility Functions
# -------------------------------
def detect_emotion(frame):
    # If the model is not available, return neutral with zero confidence
    if emotion_model is None:
        return {'emotion': 'neutral', 'confidence': 0.0}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return {'emotion': 'neutral', 'confidence': 0.0}

    emotions = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        preds = emotion_model.predict(roi, verbose=0)
        idx = np.argmax(preds)
        emotions.append((emotion_labels[idx], float(np.max(preds))))

    best_emotion, best_conf = max(emotions, key=lambda x: x[1])
    return {'emotion': best_emotion, 'confidence': best_conf}

def classify_scene(image):
    inputs = clip_processor(text=scene_labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    pred_idx = probs.argmax().item()
    return scene_labels[pred_idx]

def get_auto_location():
    try:
        data = requests.get('https://ipinfo.io/json').json()
        loc = data.get("loc", "11.0168,76.9558")  # fallback Coimbatore
        lat, lon = map(float, loc.split(","))
        return {"lat": lat, "lon": lon, "city": data.get("city", "Coimbatore"),
                "region": data.get("region", "Tamil Nadu"), "country": data.get("country", "IN")}
    except:
        return {"lat": 11.0168, "lon": 76.9558, "city": "Coimbatore",
                "region": "Tamil Nadu", "country": "IN"}

def get_weather(api_key, location):
    try:
        lat, lon = location["lat"], location["lon"]
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        data = requests.get(url).json()
        main = data["weather"][0]["main"].lower()
        temp = data["main"]["temp"]
        return {"climate": main, "temp": temp}
    except:
        return {"climate": "clear", "temp": 25}

def predict_image_weather(image):
    scene_labels_weather = ["rain", "clear", "night", "party", "nature", "street", "indoor", "outdoor", "beach", "mountain", "office", "home"]
    inputs = clip_processor(text=scene_labels_weather, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    pred_idx = probs.argmax().item()
    return scene_labels_weather[pred_idx]


# Recommend songs based on user context fields using a sophisticated scoring system
def recommend_songs_from_csv(user_emotion, user_context, user_climate, user_weather, user_location, n=8):
    df = songs_df.copy()
    preferred_language = get_language_from_region(user_location)
    
    scores = []
    for _, row in df.iterrows():
        # Base score starts at 0
        score = 0
        
        # 1. Language Relevance (0-30 points)
        if row['language'] == preferred_language:
            score += 30  # Perfect language match
        elif row['language'] == "English":
            score += 15  # English as universal second language
        
        # 2. Emotional Match (0-25 points)
        if row['emotion'].lower() == user_emotion.lower():
            score += 25  # Perfect emotion match
        
        # 3. Context Match (0-20 points)
        if row['context'].lower() == user_context.lower():
            score += 20  # Perfect context match
        
        # 4. Environmental Factors (0-25 points combined)
        if row['image_climate'].lower() == user_climate.lower():
            score += 25  # Climate match from image and scene
            
        scores.append(score)
    
    df['score'] = scores
    
    # Sort by score and get top n recommendations
    recommended = df.sort_values(by='score', ascending=False)
    
    # First, get songs in preferred language with high scores
    primary_recs = recommended[recommended['language'] == preferred_language].head(n//2)
    
    # Then, get other high-scoring songs regardless of language
    remaining_slots = n - len(primary_recs)
    other_recs = recommended[recommended['language'] != preferred_language].head(remaining_slots)
    
    # Combine and return final recommendations
    final_recs = pd.concat([primary_recs, other_recs])
    return final_recs
    scores.append(score)
    df['score'] = scores
    recommended = df.sort_values(by='score', ascending=False)
    # Only return songs with score > 0
    recommended = recommended[recommended['score'] > 0]
    if recommended.empty:
        # If no songs match the criteria, return top 8 songs
        return songs_df.head(n)
    return recommended.head(n)
    
    # Get available genres from Spotify
    try:
        available_genres = sp.recommendation_genres()
    except:
        available_genres = ["pop", "rock", "hip-hop", "electronic", "classical", "ambient", 
                          "indie", "jazz", "folk", "chill", "dance", "alternative"]
    
    # Scene-based genre mapping (using only valid Spotify genres)
    scene_genres = {
        "party": ["dance", "electronic", "pop"],
        "nature": ["ambient", "folk", "acoustic"],
        "beach": ["reggae", "pop", "tropical"],
        "concert": ["rock", "pop", "alternative"],
        "home": ["acoustic", "indie", "chill"],
        "office": ["ambient", "classical", "electronic"],
        "night": ["electronic", "chill", "alternative"]
    }
    
    # Climate-based genre additions (using only valid Spotify genres)
    climate_genres = {
        "rain": ["acoustic", "ambient", "classical"],
        "clear": ["pop", "dance", "electronic"],
        "cloudy": ["indie", "alternative", "chill"],
        "snow": ["classical", "ambient", "acoustic"]
    }
    
    # Get base features from emotion
    features = emotion_features.get(emotion.lower(), emotion_features["neutral"])
    
    # Collect genres based on scene and climate
    seed_genres = []
    if scene.lower() in scene_genres:
        seed_genres.extend([g for g in scene_genres[scene.lower()] if g in available_genres])
    if climate.lower() in climate_genres:
        seed_genres.extend([g for g in climate_genres[climate.lower()] if g in available_genres])
    
    # Ensure we have at least one valid genre
    if not seed_genres:
        seed_genres = ["pop"]
    
    # Remove duplicates and ensure all genres are valid
    seed_genres = list(dict.fromkeys([g for g in seed_genres if g in available_genres]))[:5]
    
    try:
        # Get base recommendations from emotion
        base_features = emotion_features.get(emotion.lower(), emotion_features["neutral"])
        base_genres = base_features["seed_genres"]
        
        # Search for tracks using Spotify's recommendations API
        try:
            recommendations = sp.recommendations(
                seed_genres=base_genres,
                target_valence=base_features["valence"],
                target_energy=base_features["energy"],
                limit=n
            )
        except:
            # Fallback to simpler recommendation if the first attempt fails
            recommendations = sp.recommendations(
                seed_genres=["pop"],
                limit=n
            )
        
        # Process recommendations
        songs = []
        for track in recommendations["tracks"]:
            song = {
                "track_name": track["name"],
                "artist_name": track["artists"][0]["name"],
                "spotify_url": track["external_urls"]["spotify"],
                "preview_url": track["preview_url"],
                "album_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None
            }
            songs.append(song)
        
        return songs
        
    except Exception as e:
        st.error(f"Error fetching Spotify recommendations: {str(e)}")
        return []

# -------------------------------
# Upload and Recommendations Section
# -------------------------------
st.markdown("""
    <div class="upload-section">
        <h2 style='text-align: center;'>🎯 Get Your Personalized Recommendations</h2>
        <p style='text-align: center;'>Upload an image or enter a prompt and we'll suggest the perfect soundtrack for your mood.</p>
    </div>
""", unsafe_allow_html=True)

# Custom file uploader styling
st.markdown("""
    <style>
        .uploadSection {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 2rem;
            margin: 2rem 0;
            text-align: center;
            border: 2px dashed #dee2e6;
        }
        .stFileUploader > div > div {
            padding: 2rem;
            background: white;
            border-radius: 10px;
            border: 2px dashed #dee2e6;
            margin-top: 1rem;
        }
        .stFileUploader > div > div:hover {
            background: #f8f9fa;
            border-color: #1DB954;
        }
    </style>
""", unsafe_allow_html=True)

# Centered file uploader with custom text
col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="main_uploader",
        help="Drag and drop your image here • Limit 200MB per file • JPG, JPEG, PNG formats accepted"
    )

if uploaded_file:
    # Process the uploaded image and show results
    # (existing image processing code here)
    pass

# Process uploaded image and show results
if uploaded_file is not None:
    # Process the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run analysis
    emotion_result = detect_emotion(frame)
    emotion = emotion_result['emotion']
    location = get_auto_location()
    weather_result = get_weather("e37720b4cbebdd5c37df9ae780b8688d", location)
    user_climate = weather_result['climate']
    scene_result = classify_scene(img)
    image_climate = predict_image_weather(img)

    # Display results container
    st.markdown("""
        <div style="background: linear-gradient(90deg,#08110f,#10221d); 
             border: 1px solid rgba(6,78,59,0.2); 
             border-radius: 12px; 
             padding: 2rem; 
             margin: 2rem 0;">
    """, unsafe_allow_html=True)
    
    # Create columns for the main content
    image_col, results_col = st.columns([1, 2])
    
    with image_col:
        st.image(img, caption="Your Image", use_container_width=True)

    with results_col:
        # Analysis Results
        st.subheader("📊 Analysis Results")
        
        # Emotion with confidence
        st.write("**Detected Emotion:**")
        emotion_conf = emotion_result['confidence'] * 100
        st.progress(emotion_conf / 100)
        st.text(f"{emotion.title()} ({emotion_conf:.1f}% confidence)")
        
        # Scene and Climate in two columns
        context_cols = st.columns(2)
        with context_cols[0]:
            st.write("**Scene Context (Image):**")
            st.info(scene_result.title())
            st.write("**Image Climate:**")
            st.info(image_climate.title())
        
        with context_cols[1]:
            st.write("**Your Location:**")
            st.info(f"{location.get('city')}, {location.get('region')}, {location.get('country')}")
            st.write("**Local Weather:**")
            # Create two small columns for climate and temp so they appear in one row
            wcol1, wcol2 = st.columns([1,1])
            with wcol1:
                st.metric("Climate", user_climate.title())
            with wcol2:
                st.metric("Temp (°C)", f"{weather_result.get('temp', 'N/A')}")

    # Music Recommendations Section
    st.markdown("### 🎵 Recommended Songs")
    st.markdown("Based on your mood, scene, climate, weather, and location:")

    # Get recommendations from CSV
    recommended_songs = recommend_songs_from_csv(
        user_emotion=emotion,
        user_context=scene_result,
        user_climate=image_climate,
        user_weather=user_climate,
        user_location=location,
        n=8
    )
    
    # Display recommendation factors
    st.sidebar.markdown("### 📊 Recommendation Factors")
    st.sidebar.markdown(f"""
        - **Language Priority**: {get_language_from_region(location)}
        - **Emotion**: {emotion.title()}
        - **Scene Context**: {scene_result}
        - **Image Climate**: {image_climate}
        - **Local Weather**: {user_climate}
    """)

    # Display recommendations in a 4-column grid
    cols = st.columns(4)
    for idx, (_, song) in enumerate(recommended_songs.iterrows()):
        col_idx = idx % 4
        with cols[col_idx]:
            cover = f"https://picsum.photos/seed/{song['songname'].replace(' ','')}/400/400"
            st.markdown(f"""
                <div class="song-card">
                    <img class="song-cover" src="{cover}" alt="cover" />
                    <div class="song-info">
                        <div class="song-title">{song['songname']}</div>
                        <div class="song-artist">by {song['artist']}</div>
                        <div class="song-genre">language: {song['language']}</div>
                        <div class="song-actions">
                            <div class="btn btn-play">▶</div>
                            <div class="btn btn-like">♡ </div>
                            <div class="btn btn-favorite">★ </div>
                            <div class="btn btn-preview">▶ </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# -------------------------------
# Features Section
# -------------------------------
st.markdown("""
    <h2 style='text-align: center; margin: 3rem 0 2rem 0;'>✨ Key Features</h2>
""", unsafe_allow_html=True)

feature_cols = st.columns(4)
features = [
    {
        "icon": "🎭",
        "title": "Emotion Detection",
        "desc": "Advanced AI identifies emotions from facial expressions"
    },
    {
        "icon": "🌆",
        "title": "Scene Analysis",
        "desc": "Contextual understanding of the environment"
    },
    {
        "icon": "🎵",
        "title": "Smart Recommendations",
        "desc": "Personalized music suggestions based on mood"
    },
    {
        "icon": "🌤️",
        "title": "Climate Context",
        "desc": "Weather-aware music recommendations"
    }
]

for col, feature in zip(feature_cols, features):
    with col:
        st.markdown(f"""
            <div class="feature-card">
                <h1 style='text-align: center;'>{feature['icon']}</h1>
                <h4 style='text-align: center;'>{feature['title']}</h4>
                <p style='text-align: center;'>{feature['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------
# How It Works Section
# -------------------------------
st.markdown("""
    <div class="how-it-works" style="text-align: center; max-width: 800px; margin: 0 auto;">
        <h2>🔍 How It Works</h2>
        <br>
        <div style="display: inline-block; text-align: left;">
            <ol style="list-style-position: inside;">
                <li><strong>Upload Your Image:</strong> Share a photo that captures your moment</li>
                <li><strong>AI Analysis:</strong> Our advanced AI analyzes emotions, scene, and context</li>
                <li><strong>Emotion Detection:</strong> Identifies your emotional state from facial expressions</li>
                <li><strong>Scene Recognition:</strong> Understands the environment and setting</li>
                <li><strong>Climate Context:</strong> Considers weather and atmospheric conditions</li>
                <li><strong>Music Magic:</strong> Generates perfectly matched song recommendations</li>
            </ol>
        </div>
        <br>
        <p><em>Experience the future of mood-based music recommendations!</em></p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# Footer Section
# -------------------------------
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid rgba(6,78,59,0.2);">
        <h3>🎵 Muzly Music Recommender</h3>
        <p>Crafted with ❤️ for music lovers</p>
        <p>© 2025 Muzly. All rights reserved.</p>
        <small>Powered by Advanced AI & Music Analysis</small>
    </div>
""", unsafe_allow_html=True)

# Show emotion model warning after file upload section
if emotion_model is None:
    st.warning("Emotion detection model not found — emotion analysis will be disabled.\n\nPlace 'best_emotion_model.h5' in the 'model' directory to enable it.")

# Process uploaded file (handled in results section below)

    # Create columns for the main content
    image_col, results_col = st.columns([1, 2])
    
    with image_col:
        st.image(img, caption="Your Image", use_container_width=True)

    with results_col:
        # Analysis Results
        st.subheader("📊 Analysis Results")
        
        # Emotion with confidence
        st.write("**Detected Emotion:**")
        emotion_conf = emotion_result['confidence'] * 100
        st.progress(emotion_conf / 100)
        st.text(f"{emotion.title()} ({emotion_conf:.1f}% confidence)")
        
        # Scene and Climate in two columns
        context_cols = st.columns(2)
        with context_cols[0]:
            st.write("**Scene Context:**")
            st.info(scene_result.title())
        
        with context_cols[1]:
            st.write("**Weather Context:**")
            st.metric("Location", user_climate.title())
            st.metric("Image Climate", image_climate.title())

    # End of file
