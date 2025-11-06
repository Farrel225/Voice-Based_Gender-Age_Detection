# --- t1.py (Memory-Efficient Version with MP3 FIX) ---
import pandas as pd
import os
import librosa
import numpy as np
import time
import shutil # For safely creating the output folder
from pydub import AudioSegment # NEW: Import pydub
import tempfile # NEW: Import tempfile

# --- Configuration ---
METADATA_FILE = r"C:\Users\Test\Downloads\Datasets\cv-valid-train.csv"
CLIPS_FOLDER = r"C:\Users\Test\Downloads\Datasets\cv-valid-train" 
OUTPUT_FEATURES_DIR = "processed_data" # Folder to store individual .npy files
LABELS_OUTPUT_FILE = "final_labels.csv" # The master answer key

# --- Audio Processing Parameters ---
TARGET_SR = 22050  # Sample rate
N_MFCC = 40        # Number of MFCC features
MAX_LEN_SECONDS = 5 # Pad/truncate all clips to 5 seconds
TARGET_SAMPLES = TARGET_SR * MAX_LEN_SECONDS # Calculate total samples

# --- 1. Setup Output Directory ---
print(f"🔄 Setting up output directory: {OUTPUT_FEATURES_DIR}")
# If the script failed, delete the folder to restart cleanly
if os.path.exists(OUTPUT_FEATURES_DIR):
    print(f"   WARNING: Output directory '{OUTPUT_FEATURES_DIR}' already exists. Deleting it to start fresh.")
    shutil.rmtree(OUTPUT_FEATURES_DIR)
os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
print(f"   Output directory is ready at: {os.path.abspath(OUTPUT_FEATURES_DIR)}")


# --- 2. Load Metadata ---
print(f"🔄 Loading metadata from {METADATA_FILE}...")
try:
    df = pd.read_csv(METADATA_FILE, sep=',', on_bad_lines='skip') 
except FileNotFoundError:
    print(f"❌ ERROR: Metadata file not found at '{METADATA_FILE}'.")
    exit()
except Exception as e:
    print(f"❌ ERROR: Could not load metadata file: {e}")
    exit()

print(f"   Original file count: {len(df)}")
print(f"   Columns found: {list(df.columns)}")

# --- 3. Clean Metadata ---
try:
    df_clean = df.dropna(subset=['gender', 'age', 'filename'])
except KeyError:
    print("❌ ERROR: CSV missing 'gender', 'age', or 'filename' columns.")
    exit()

df_clean.loc[:, 'gender'] = df_clean['gender'].astype(str).apply(lambda x: x.split('_')[0])
df_clean = df_clean[df_clean['gender'].isin(['male', 'female'])]
df_clean = df_clean[df_clean['filename'].notna()]

print(f"   Cleaned file count (with valid labels): {len(df_clean)}")
if len(df_clean) == 0:
    print("❌ ERROR: No valid data found after cleaning.")
    exit()
print("-" * 30)

# --- 4. Feature Extraction Loop (Saving one file at a time) ---
print("🎶 Starting audio processing and feature extraction...")
print(f"   This will process {len(df_clean)} files. This will take a very long time!")
start_time = time.time()

processed_count = 0
skipped_count = 0
error_count = 0
new_labels_list = [] # We will build our new answer key here

for index, row in df_clean.iterrows():
    filename = str(row['filename'])
    full_audio_path = os.path.join(CLIPS_FOLDER, filename)
    wav_path_to_load = None # Path to the file librosa will load
    needs_cleanup = False   # Flag to delete temp file

    if not os.path.exists(full_audio_path):
        full_audio_path = os.path.join(CLIPS_FOLDER, "clips", filename)
        if not os.path.exists(full_audio_path):
            skipped_count += 1
            continue

    try:
        # --- NEW MP3 LOADING LOGIC ---
        _, ext = os.path.splitext(full_audio_path)
        
        if ext.lower() == '.mp3':
            audio = AudioSegment.from_mp3(full_audio_path)
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                audio.export(temp_wav.name, format="wav")
                wav_path_to_load = temp_wav.name
                needs_cleanup = True
        elif ext.lower() == '.wav':
            wav_path_to_load = full_audio_path
            needs_cleanup = False
        else:
            # Skip unsupported formats like .txt etc.
            skipped_count += 1
            continue
        # --- END NEW LOADING LOGIC ---

        # Load the WAV file (or original WAV) with Librosa
        y, sr = librosa.load(wav_path_to_load, sr=TARGET_SR)
        
        # Clean up the temp file if one was created
        if needs_cleanup:
            os.remove(wav_path_to_load)
        
        # --- Pad or Truncate to fixed length (TARGET_SAMPLES) ---
        if len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES] # Truncate long files
        elif len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant') # Pad short files
            
        # Normalize audio volume (fixes "feeble" audio)
        y_normalized = librosa.util.normalize(y)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=N_MFCC)

        # Transpose: (n_mfcc, time) -> (time, n_mfcc)
        mfccs = mfccs.T
        
        # --- Save as individual .npy file ---
        npy_filename = f"{os.path.splitext(filename)[0]}.npy"
        save_path = os.path.join(OUTPUT_FEATURES_DIR, npy_filename)
        
        np.save(save_path, mfccs.astype(np.float32)) 
        
        new_labels_list.append([npy_filename, row['gender'], row['age']])
        processed_count += 1
        
        if processed_count % 500 == 0 or processed_count == len(df_clean):
            elapsed = time.time() - start_time
            print(f"   Processed {processed_count}/{len(df_clean)} files... (Total time: {elapsed/60:.2f} mins)")

    except Exception as e:
        error_count += 1
        # Clean up temp file if error happened after creation
        if needs_cleanup and 'wav_path_to_load' in locals() and os.path.exists(wav_path_to_load):
             os.remove(wav_path_to_load)

print("-" * 30)
processing_time = time.time() - start_time
print(f"✅ Feature extraction finished in {processing_time / 60:.2f} minutes (or {processing_time / 3600:.2f} hours).")
print(f"   Successfully processed and saved: {processed_count} files")
print(f"   Skipped (not found or wrong format): {skipped_count} files")
print(f"   Errors during processing: {error_count} files")

# --- 5. Save New Master Answer Key ---
if not new_labels_list:
    print("❌ ERROR: No files were processed. No labels to save.")
    exit()
    
print(f"\n💾 Saving new master label file to {LABELS_OUTPUT_FILE}...")
labels_df_final = pd.DataFrame(new_labels_list, columns=['npy_filename', 'gender', 'age'])
labels_df_final.to_csv(LABELS_OUTPUT_FILE, index=False)

print("\n🏁 Preprocessing Script Done!")
print(f"   You now have a folder '{OUTPUT_FEATURES_DIR}' with {processed_count} .npy files.")
print(f"   And a new answer key: {LABELS_OUTPUT_FILE}")
print("\n➡️ You are now ready to run training.py")
