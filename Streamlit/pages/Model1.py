import streamlit as st
import numpy as np
import librosa

def process_audio(file):
    import time
    start_time = time.time()

    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    max_pad_len = 100
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc_flat = mfcc.flatten().reshape(1, -1)
    lda_features = np.dot(mfcc_flat, st.session_state.lda_eigenvectors)

    predicted_label = st.session_state.nb_classifier.predict(lda_features)[0]
    speaker_name = st.session_state.inv_label_map[predicted_label]

    return speaker_name


def main():
    if 'resources_loaded' not in st.session_state:
        st.error("Please start from the Home page to load the model first!")
        return

    st.title("Speaker Recognition using Naive Bayes with LDA")
    st.write("Upload a `.wav` file to classify it using the trained model.")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        speaker_name = process_audio(uploaded_file)
        st.write(f"Speaker Name: {speaker_name}")


if __name__ == "__main__":
    main()

