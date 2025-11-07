# 🎙️Voice-Based Gender & Age Detection using CNN & RNN
This document contains a deep learning project that classifies a speaker's gender and age group from a raw audio file.

📝 Project Description
The goal of this project was to build, train, and deploy a complete deep learning application. The final product is an interactive web app that can accept either a live 5-second microphone recording or an uploaded audio file (.mp3 or .wav) and instantly predict the speaker's gender and age category.

The core of the project is a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).
- The CNN scans the audio's spectrogram (an image of the sound) to find spatial features and patterns.
- The RNN then analyzes the sequence of those patterns over time to understand the temporal context of the speech.

The model was trained on the Common Voice dataset from Kaggle, using over 72,000+ cleaned audio clips (totaling over 400 hours of validated speech) to ensure high accuracy.

🚀 Application Demo
Here is the final Streamlit application in action, successfully predicting the gender and age group from an uploaded audio file.

<img width="634" height="874" alt="image" src="https://github.com/user-attachments/assets/f65c7431-2ac0-4319-acb2-9dc76330fe7c" />

🛠️ Tools, Libraries & Workflow
This project was built in three main parts: preprocessing, training, and deployment.

**1. Preprocessing (t1.py)**

<img width="1108" height="695" alt="image" src="https://github.com/user-attachments/assets/38a1a532-cfe7-435c-b15f-3c932c3809bb" />
<img width="813" height="714" alt="image" src="https://github.com/user-attachments/assets/6c238f7b-d9c3-40d3-852a-b2433f49bd02" />

  - Kaggle Notebooks: The pre-processing took a grand total of almost 5 hours (specifically 283 minute as per the pic) on the Kaggle cloud-based environment to process the massive 72,000+ file dataset without crashing a local machine.
  - Pandas: Used to load and clean the master cv-valid-train.csv "answer key" (metadata).
  - Pydub (& ffmpeg): Used to load all .mp3 files from the dataset and convert them to a standard .wav format.
  - Librosa: Used for all audio processing:
      Loading audio with a uniform sample rate (22050Hz).
      Padding or truncating all clips to a fixed 5-second length.
      Normalizing audio to fix "feeble" or quiet recordings.
      Extracting MFCC (Mel-Frequency Cepstral Coefficients) features.
      NumPy: Used to save each processed file as a separate, small .npy file to create a memory-efficient "data generator" pipeline.

**2. Model Training (training.py)**

<img width="538" height="702" alt="image" src="https://github.com/user-attachments/assets/983ec630-866e-40f2-9e54-3b616db5a08b" />
<img width="1135" height="725" alt="image" src="https://github.com/user-attachments/assets/c2e19a4c-1724-4853-b0c1-3419856359bf" />

  TensorFlow (Keras): Used to:
    - Define the DataGenerator to feed the 72,000+ .npy files to the model in small batches (to save RAM).
    - Build the hybrid CNN+RNN model architecture & train the model.
    - Use ModelCheckpoint and EarlyStopping callbacks to save the best-performing model (best_voice_model.h5).
    - scikit-learn: Used to LabelEncoder (to convert text labels like 'female' or 'twenties' into numbers) and to calculate class_weight to handle the imbalanced dataset (e.g., more '20s' voices than '60s').
      
**3. Main Application (Main.py)**

  Kaggle Notebooks (GPU): Trained the model using a free NVIDIA P100 GPU to complete the training in ~1-3 hours instead of 20-40+ hours.
  
  - Streamlit: Used to build the entire interactive web application and user interface.
  - SpeechRecognition: Used to capture live audio from the user's microphone for the "Speak & Analyze" feature.
  - Librosa/Pydub/TensorFlow: Used within the app to run the exact same preprocessing (MP3 conversion, 5-second padding, normalization, MFCCs) on new, incoming audio before feeding it to the loaded best_voice_model.h5 for a prediction.
  
<img width="896" height="768" alt="image" src="https://github.com/user-attachments/assets/b4da7999-d2de-4c5c-b488-72859af2f4a2" />
