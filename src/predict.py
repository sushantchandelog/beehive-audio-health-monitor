import os
import sys
import librosa
import numpy as np
import joblib

def predict_hive_health(audio_path, model_path="models/hive_model.pkl"):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    # Load trained model
    model = joblib.load(model_path)

    # Load audio & extract features (22.05 kHz)
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Predict class & confidence
    prediction = model.predict(mfccs_mean)[0]
    probabilities = model.predict_proba(mfccs_mean)[0]

    status = "QueenBee Present (Healthy)" if prediction == 1 else "QueenBee Absent (Distressed)"
    confidence = probabilities[prediction] * 100

    print("\n=================================")
    print("  BEEHIVE HEALTH DIAGNOSTIC")
    print("=================================")
    print(f"Audio File : {os.path.basename(audio_path)}")
    print(f"Status     : {status}")
    print(f"Confidence : {confidence:.2f}%\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default test path (adjust to match an existing wav file)
        test_file = "data/raw/QueenBee Present/QueenBee_1.wav"

    predict_hive_health(test_file)