# --- training.py ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import joblib # For saving encoders

## Load Data
# --------------------------------------------------------------------------
print("🔄 Loading preprocessed data...")
try:
    X = np.load('X_data.npy')
    labels_df = pd.read_csv('Y_labels.csv')
except FileNotFoundError:
    print("❌ ERROR: X_data.npy or Y_labels.csv not found.")
    print("   Please run t1.py first.")
    exit()

print(f"   Features loaded with shape: {X.shape}")
print(f"   Labels loaded (first 5 rows):\n{labels_df.head()}")

## Prepare Labels
# --------------------------------------------------------------------------
# --- Encode Gender ---
gender_encoder = LabelEncoder()
y_gender = gender_encoder.fit_transform(labels_df['gender'])
y_gender_onehot = to_categorical(y_gender, num_classes=len(gender_encoder.classes_))

# --- Encode Age (Automatic Bucket Detection) ---
y_age_labels = labels_df['age'].astype(str)
age_encoder = LabelEncoder()
y_age = age_encoder.fit_transform(y_age_labels)
num_age_classes = len(age_encoder.classes_)
y_age_onehot = to_categorical(y_age, num_classes=num_age_classes)

print("\n✅ Labels successfully one-hot encoded.")
print(f"   Gender encoding: {gender_encoder.classes_}")
print(f"   Age encoding: {age_encoder.classes_}")
print(f"   Found {num_age_classes} age classes.")

## Split Data
# --------------------------------------------------------------------------
print("\nSplitting data into training and validation sets...")
X_train, X_val, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(
    X, y_gender_onehot, y_age_onehot,
    test_size=0.2,          # 20% for validation
    random_state=42,        # For reproducible splits
    stratify=y_age_onehot  # Keep age distribution similar in both sets
)
print(f"   X_train shape: {X_train.shape}")
print(f"   X_val shape: {X_val.shape}")

## Build Model
# --------------------------------------------------------------------------
print("\nBuilding the CNN + LSTM model...")
input_shape = (X_train.shape[1], X_train.shape[2]) # (max_length, n_mfcc)

input_layer = Input(shape=input_shape, name='input_layer')

# CNN Layers
x = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_1')(input_layer)
x = BatchNormalization(name='batchnorm_1')(x)
x = MaxPooling1D(pool_size=2, name='maxpool_1')(x)

x = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_2')(x)
x = BatchNormalization(name='batchnorm_2')(x)
x = MaxPooling1D(pool_size=2, name='maxpool_2')(x)

x = Conv1D(256, kernel_size=3, activation='relu', padding='same', name='conv1d_3')(x)
x = BatchNormalization(name='batchnorm_3')(x)
x = MaxPooling1D(pool_size=2, name='maxpool_3')(x)

# LSTM Layer
x = LSTM(128, return_sequences=False, name='lstm_1')(x)
x = Dropout(0.5, name='dropout_1')(x)

# Output Heads
gender_output = Dense(len(gender_encoder.classes_), activation='softmax', name='gender_output')(x)
age_output = Dense(num_age_classes, activation='softmax', name='age_output')(x)

# Create and Compile Model
model = Model(inputs=input_layer, outputs=[gender_output, age_output], name='Voice_GenderAge_CNN_LSTM')

model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'age_output': 'accuracy'})

model.summary()

## Define Callbacks
# --------------------------------------------------------------------------
output_dir = "."
best_model_path = os.path.join(output_dir, 'best_voice_model.h5')

model_checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    save_best_only=True,
    monitor='val_loss', # Monitor total validation loss
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop after 10 epochs with no improvement
    mode='min',
    verbose=1,
    restore_best_weights=True # Restore the best weights at the end
)

## Train Model
# --------------------------------------------------------------------------
print("\n🚀 Starting model training... (This will use your GPU!)")
EPOCHS = 100 # Set high, EarlyStopping will find the best one
BATCH_SIZE = 32 # 32 is a good default for GPU training

history = model.fit(
    X_train,
    {'gender_output': y_gender_train, 'age_output': y_age_train},
    validation_data=(X_val, {'gender_output': y_gender_val, 'age_output': y_age_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]
)

print("\n✅ Training complete!")

## Save Encoders
# --------------------------------------------------------------------------
print("\n💾 Saving label encoders...")
joblib.dump(gender_encoder, os.path.join(output_dir, 'gender_encoder.joblib'))
joblib.dump(age_encoder, os.path.join(output_dir, 'age_encoder.joblib'))
print("   Encoders saved as .joblib files.")

print(f"\nAll done! Best model saved to: {best_model_path}")
