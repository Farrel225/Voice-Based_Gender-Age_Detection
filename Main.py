# --- Main.py ---
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import speech_recognition as sr
import os
import tempfile
from sklearn.preprocessing import LabelEncoder
import traceback
import math

# --- Configuration (MUST MATCH t1.py and training.py) ---
MODEL_PATH = "best_voice_model.h5"

# Audio parameters
TARGET_SR = 22050
N_MFCC = 40
MAX_LEN_SECONDS = 5
TARGET_SAMPLES = TARGET_SR * MAX_LEN_SECONDS
HOP_LENGTH = 512 # Default Librosa hop_length
N_TIMESTEPS = math.ceil((TARGET_SR * MAX_LEN_SECONDS) / HOP_LENGTH) # Should be 216

# --- 1. Load Model (once, cached) ---
@st.cache_resource # Cache the model and encoders
def load_model_and_encoders():
    print("Loading model and encoders...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # --- ⬇️ CRITICAL CHANGE APPLIED HERE ⬇️ ---
        # These lists now match your Kaggle training output exactly.
        GENDER_CLASSES = ['female', 'male'] 
        AGE_CLASSES = ['eighties', 'fifties', 'fourties', 'seventies', 'sixties', 'teens', 'thirties', 'twenties']
        # --- ⬆️ CRITICAL CHANGE APPLIED HERE ⬆️ ---

        gender_encoder = LabelEncoder()
        gender_encoder.fit(GENDER_CLASSES)
        
        age_encoder = LabelEncoder()
        age_encoder.fit(AGE_CLASSES)
        
        print("Model and encoders loaded successfully.")
        return model, gender_encoder, age_encoder
        
    except FileNotFoundError:
        st.error(f"❌ ERROR: Model file not found at '{MODEL_PATH}'.")
        st.error("Please ensure the trained 'best_voice_model.h5' file is in the same directory as this app.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(traceback.format_exc())
        return None, None, None

model, gender_encoder, age_encoder = load_model_and_encoders()

# --- 3. Preprocessing Function (Mirrors t1.py) ---
def analyze_audio(audio_file_path):
    """
    Processes a single audio file, applies the SAME 5-second
    preprocessing as training, and predicts gender/age.
    """
    if model is None:
        st.error("Model not loaded. Cannot analyze.")
        return None
        
    wav_path_to_load = None
    needs_cleanup = False

    try:
        # --- Load and Convert MP3/WAV (same as t1.py) ---
        _, ext = os.path.splitext(audio_file_path)
        
        if ext.lower() == '.mp3':
            audio = AudioSegment.from_mp3(audio_file_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                audio.export(temp_wav.name, format="wav")
                wav_path_to_load = temp_wav.name
                needs_cleanup = True
        elif ext.lower() == '.wav':
            wav_path_to_load = audio_file_path
            needs_cleanup = False
        else:
             st.error(f"Unsupported file format: {ext}. Please use WAV or MP3.")
             return None

        y, sr = librosa.load(wav_path_to_load, sr=TARGET_SR)
        
        if needs_cleanup and os.path.exists(wav_path_to_load):
            os.remove(wav_path_to_load)

        # --- Pad or Truncate (same as t1.py) ---
        if len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')
            
        # --- Normalize, Extract Features, Transpose (same as t1.py) ---
        y_normalized = librosa.util.normalize(y)
        mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=N_MFCC)
        mfccs = mfccs.T
        
        # --- Reshape for Model (add batch dimension) ---
        if mfccs.shape[0] != N_TIMESTEPS:
            # Pad/trim the feature array itself to the exact N_TIMESTEPS
            if mfccs.shape[0] > N_TIMESTEPS:
                 mfccs = mfccs[:N_TIMESTEPS, :]
            else:
                 mfccs = np.pad(mfccs, ((0, N_TIMESTEPS - mfccs.shape[0]), (0,0)), mode='constant')

        features = np.expand_dims(mfccs, axis=0) # Shape -> (1, N_TIMESTEPS, N_MFCC)

        # --- Make Prediction ---
        predictions = model.predict(features)
        
        gender_prediction_probs = predictions[0][0]
        age_prediction_probs = predictions[1][0]

        # --- Decode Predictions ---
        predicted_gender_index = np.argmax(gender_prediction_probs)
        predicted_gender = gender_encoder.inverse_transform([predicted_gender_index])[0]
        gender_confidence = gender_prediction_probs[predicted_gender_index] * 100

        predicted_age_index = np.argmax(age_prediction_probs)
        predicted_age_bucket = age_encoder.inverse_transform([predicted_age_index])[0]
        age_confidence = age_prediction_probs[predicted_age_index] * 100

        return {
            "gender": predicted_gender.capitalize(),
            "age": predicted_age_bucket.capitalize(),
            "gender_confidence": f"{gender_confidence:.2f}%",
            "age_confidence": f"{age_confidence:.2f}%"
        }

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.error(traceback.format_exc())
        if needs_cleanup and 'wav_path_to_load' in locals() and os.path.exists(wav_path_to_load):
             os.remove(wav_path_to_load)
        return None

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Voice-Based Gender/Age Detection",
    page_icon="🎤",
    layout="centered"
)

# --- Custom CSS (Same as before) ---
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #2c3e50, #3498db);
    color: white;
    padding: 2rem;
}
.stApp > header { background-color: transparent; }
.stButton>button {
    background-color: #87CEFA; color: #2c3e50; border: 2px solid #16a085;
    border-radius: 8px; padding: 10px 24px; font-size: 16px; font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.2s ease; cursor: pointer;
    margin-top: 5px;
}
.stButton>button:hover {
    background-color: #4682B4; color: white; border-color: #1abc9c;
    box-shadow: 0 6px 8px rgba(0,0,0,0.15); transform: translateY(-2px);
}
.title-text {
    font-size: 32px; font-weight: bold; color: #ecf0f1;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4); text-align: center; margin-bottom: 20px;
}
.stFileUploader {
    background-color: rgba(255, 255, 255, 0.08); border-radius: 8px;
    padding: 15px; border: 2px dashed #1abc9c;
}
.stAudio { width: 100%; margin-top: 15px; margin-bottom: 15px; }
h3 { color: #9cd3f3; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 5px; margin-top: 1.5rem; }
.result-text strong { color: #87CEFA; }
.result-text { font-size: 1.1em; line-height: 1.7; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Main Application UI ---
st.markdown("<h1 class='title-text'>🎙️ Voice-Based Gender/Age Detection</h1>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

if model is not None:
    st.success("🧠 Model loaded successfully!")
else:
    st.error("🚨 Model failed to load. The application cannot proceed.")
    st.stop() 

st.markdown("Analyze a voice recording or upload an audio file to predict the speaker's **gender** and **age category**.")

col1, col2 = st.columns(2, gap="large")

# --- Column 1: Live Recording ---
with col1:
    st.subheader("1. Speak & Analyze")
    st.caption("Record your voice directly using your microphone.")

    if st.button("🎤 Start Recording (5s)", key="record_button"):
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        temp_rec_path = None

        try:
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=TARGET_SR) as source:
                status_placeholder.info("🎙️ Listening... Speak clearly for 5 seconds.")
                audio_data = r.listen(source, timeout=5, phrase_time_limit=5)
            status_placeholder.success("✅ Recording complete! Analyzing...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_rec:
                temp_rec.write(audio_data.get_wav_data())
                temp_rec_path = temp_rec.name

            results = analyze_audio(temp_rec_path)

            status_placeholder.empty()
            if results:
                results_placeholder.markdown(f"""
                <div class="result-text">
                    <h3>Results (Live Recording):</h3>
                    <p><strong>Predicted Gender:</strong> {results['gender']} &nbsp; (Confidence: {results['gender_confidence']})</p>
                    <p><strong>Predicted Age Group:</strong> {results['age']} &nbsp; (Confidence: {results['age_confidence']})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                results_placeholder.warning("⚠️ Analysis could not be completed.")

        except sr.WaitTimeoutError:
            status_placeholder.warning("묵 No speech detected. Please ensure your mic is working and try again.")
        except Exception as e:
            status_placeholder.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())
        finally:
            if temp_rec_path and os.path.exists(temp_rec_path):
                 os.remove(temp_rec_path)

# --- Column 2: File Upload ---
with col2:
    st.subheader("2. Upload & Analyze")
    st.caption("Upload an audio file (WAV or MP3).")

    uploaded_file = st.file_uploader(" ", type=["wav", "mp3"], key="file_uploader", label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[-1]}')
        temp_file_path = None

        if st.button("📊 Analyze Uploaded File", key="upload_button"):
            status_placeholder_up = st.empty()
            results_placeholder_up = st.empty()

            try:
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                status_placeholder_up.info("⏳ Analyzing uploaded file...")
                results = analyze_audio(temp_file_path)

                status_placeholder_up.empty()
                if results:
                    results_placeholder_up.markdown(f"""
                    <div class="result-text">
                        <h3>Results (Uploaded File):</h3>
                        <p><strong>Predicted Gender:</strong> {results['gender']} &nbsp; (Confidence: {results['gender_confidence']})</p>
                        <p><strong>Predicted Age Group:</strong> {results['age']} &nbsp; (Confidence: {results['age_confidence']})</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    results_placeholder_up.warning("⚠️ Analysis could not be completed.")

            except Exception as e:
                 status_placeholder_up.error(f"An error occurred: {e}")
                 st.error(traceback.format_exc())
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                     os.remove(temp_file_path)

# --- Footer Separator ---
st.markdown("---", unsafe_allow_html=True)

# --- Instructions / About Section ---
with st.expander("ℹ️ About the Model & Workflow", expanded=False):
    st.markdown(f"""
    This application uses a trained **CNN + LSTM** model to predict gender and age.
    
    **Model Details:**
    * **Input Shape:** `(1, {N_TIMESTEPS}, {N_MFCC})` (1 batch, {N_TIMESTEPS} timesteps, {N_MFCC} features)
    * **Audio Length:** All audio is processed to a fixed **{MAX_LEN_SECONDS}-second** duration.
    * **Gender Classes:** `{list(gender_encoder.classes_)}`
    * **Age Classes:** `{list(age_encoder.classes_)}`
    
    **Workflow:**
    1.  **`t1.py` (Preprocess):** Loops through all {len(labels_df)} audio files. For each file, it converts from MP3, pads/truncates to {MAX_LEN_SECONDS}s, normalizes, and extracts MFCCs. Saves each as a separate `.npy` file in `{FEATURES_DIR}/` and creates `final_labels.csv`.
    2.  **`training.py` (Train):** Loads `final_labels.csv`. Uses a `DataGenerator` to feed batches of `.npy` files to the GPU for training, saving memory. Saves the `best_voice_model.h5`.
    3.  **`Main.py` (App):** Loads `best_voice_model.h5`. When you upload or record, it performs the *exact same* preprocessing steps from `t1.py` on your single file to get a prediction.
    """)
