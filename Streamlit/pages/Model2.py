import streamlit as st
import numpy as np
import librosa

st.set_page_config(page_title="Linguistix", page_icon="🎙️", layout="centered")

def process_audio(file):
    import time
    start_time = time.time()

    # Load audio and extract MFCC features
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or trim MFCC features to a fixed length
    max_pad_len = 100
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    # Flatten and project features using the loaded LDA eigenvectors
    mfcc_flat = mfcc.flatten().reshape(1, -1)
    X_features = np.dot(mfcc_flat, st.session_state.lda_eigenvectors)

    # Predict speaker using the GMM model
    predicted_label = st.session_state.gmm_model.predict(X_features)[0]
    # Use get() to avoid KeyError if the label isn't in the mapping
    speaker_name = st.session_state.inv_label_map.get(predicted_label, f"Speaker {predicted_label}")

    return speaker_name

def main():
    if 'resources_loaded' not in st.session_state:
        st.error("Please start from the Home page to load the model first!")
        return

    st.title("Speaker Recognition using GMM Clusturing")
    st.write("Upload a `.wav` file to classify it using the trained model.")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        speaker_name = process_audio(uploaded_file)
        st.write(f"Speaker Name: {speaker_name}")

if __name__ == "__main__":
    main()