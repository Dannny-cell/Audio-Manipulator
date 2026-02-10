import streamlit as st
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment

# Import your custom modules
from plotting import plot_waveform, plot_spectrum, plot_spectrogram
from filtering import apply_filter, apply_eq
from noise_reduction import reduce_noise_spectral_gating
from effects import apply_panning, apply_pitch_shift, apply_time_stretch, apply_delay

# --- Page Configuration ---
st.set_page_config(page_title="Audio Manipulator", page_icon="🎚️", layout="wide")

# --- Session State Initialization ---
# This ensures that variables persist between reruns
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.samplerate = None
    st.session_state.processed_data = None
    st.session_state.filename = None

# --- Helper Functions ---
def load_audio_from_bytes(file_bytes, filename):
    """Loads audio from uploaded file bytes, ensuring it's in stereo format."""
    try:
        # always_2d=True ensures the output is a 2D array (stereo),
        # which simplifies the rest of the code. Mono files will be duplicated into two channels.
        audio_data, samplerate = sf.read(io.BytesIO(file_bytes), dtype='float32', always_2d=True)
        st.session_state.audio_data = audio_data
        st.session_state.processed_data = audio_data.copy()
        st.session_state.samplerate = samplerate
        st.session_state.filename = filename
    except Exception as e:
        st.error(f"Error loading audio file: {e}")

# --- UI Layout ---
st.title("🎚️ Audio Manipulator Pro")
st.markdown("Upload an audio file to begin. All effects are cumulative. Reset to start over.")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'ogg'])

# Load audio only when a new file is uploaded
if uploaded_file and (st.session_state.filename != uploaded_file.name):
    with st.spinner('Loading and preparing audio...'):
        load_audio_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

# --- Main Application Body (only runs if audio is loaded) ---
if st.session_state.audio_data is not None:
    st.success(f"Loaded: `{st.session_state.filename}`")
    col1, col2 = st.columns([1, 2]) # Column for controls, column for display

    # --- Column 1: All the interactive controls ---
    with col1:
        st.header("Audio Controls")
        if st.button("Reset to Original", use_container_width=True):
            st.session_state.processed_data = st.session_state.audio_data.copy()
            st.rerun()

        with st.expander("👋 Stereo Panning", expanded=True):
            pan_value = st.slider("Pan (Left <-> Right)", -1.0, 1.0, 0.0, 0.05)
            if st.button("Apply Panning", use_container_width=True):
                st.session_state.processed_data = apply_panning(st.session_state.processed_data, st.session_state.samplerate, pan_value)
                st.rerun()

        with st.expander("⏰ Time & Pitch"):
            pitch_steps = st.slider("Pitch Shift (semitones)", -12, 12, 0)
            if st.button("Apply Pitch Shift", use_container_width=True):
                with st.spinner("Shifting pitch..."):
                    st.session_state.processed_data = apply_pitch_shift(st.session_state.processed_data, st.session_state.samplerate, pitch_steps)
                st.rerun()

            time_rate = st.slider("Time Stretch (speed)", 0.5, 2.0, 1.0, 0.05)
            if st.button("Apply Time Stretch", use_container_width=True):
                with st.spinner("Stretching time..."):
                    st.session_state.processed_data = apply_time_stretch(st.session_state.processed_data, st.session_state.samplerate, time_rate)
                st.rerun()

        with st.expander("✨ Audio Effects"):
            st.subheader("Delay (Echo)")
            delay_ms = st.slider("Delay Time (ms)", 0, 1000, 0)
            delay_feedback = st.slider("Feedback (decay)", 0.0, 0.9, 0.5, 0.05)
            delay_mix = st.slider("Wet/Dry Mix", 0.0, 1.0, 0.0, 0.05)
            if st.button("Apply Delay", use_container_width=True):
                with st.spinner("Adding delay..."):
                    st.session_state.processed_data = apply_delay(st.session_state.processed_data, st.session_state.samplerate, delay_ms, delay_feedback, delay_mix)
                st.rerun()

        with st.expander("🔊 Equalizer (3-Band)"):
            bass = st.slider("Bass (dB)", -12.0, 12.0, 0.0, 0.5)
            mid = st.slider("Mids (dB)", -12.0, 12.0, 0.0, 0.5)
            treble = st.slider("Treble (dB)", -12.0, 12.0, 0.0, 0.5)
            if st.button("Apply EQ", use_container_width=True):
                st.session_state.processed_data = apply_eq(st.session_state.processed_data, st.session_state.samplerate, bass, mid, treble)
                st.rerun()

        with st.expander("🔪 Filters"):
            f_type = st.selectbox("Filter Type", ["lowpass", "highpass", "bandpass"])
            if f_type == "bandpass":
                cutoff = st.slider("Cutoff Frequencies (Hz)", 20, 20000, (300, 3000))
            else:
                cutoff = st.slider("Cutoff Frequency (Hz)", 20, 20000, 1000)
            if st.button("Apply Filter", use_container_width=True):
                st.session_state.processed_data = apply_filter(st.session_state.processed_data, st.session_state.samplerate, cutoff, filter_type=f_type)
                st.rerun()

        with st.expander("🤫 Noise Reduction"):
            if st.button("Apply Basic Noise Reduction", use_container_width=True):
                with st.spinner("Reducing noise... (this can be slow)"):
                    st.session_state.processed_data = reduce_noise_spectral_gating(st.session_state.processed_data, st.session_state.samplerate)
                st.rerun()

        with st.expander("💾 Export Audio"):
            export_format = st.selectbox("Export Format", ["wav", "mp3", "ogg"])

            # Use a buffer to hold the processed audio in memory
            buffer = io.BytesIO()
            # Write processed data to buffer as WAV first, as it's lossless
            sf.write(buffer, st.session_state.processed_data, st.session_state.samplerate, format='WAV')
            buffer.seek(0)

            # If a different format is requested, use pydub for conversion
            if export_format != "wav":
                sound = AudioSegment.from_wav(buffer)
                final_buffer = io.BytesIO()
                sound.export(final_buffer, format=export_format)
                final_buffer.seek(0)
            else:
                final_buffer = buffer

            st.download_button(
                label=f"Download as {export_format.upper()}",
                data=final_buffer,
                file_name=f"processed_audio.{export_format}",
                mime=f"audio/{export_format}",
                use_container_width=True
            )

    # --- Column 2: Display for playback and visualizations ---
    with col2:
        st.header("Playback & Visualization")
        # Transpose the data for st.audio which expects shape (channels, samples)
        st.audio(st.session_state.processed_data.T, sample_rate=st.session_state.samplerate)

        tab1, tab2, tab3 = st.tabs(["Frequency Spectrum", "Waveform", "Spectrogram"])

        with tab1:
            st.pyplot(plot_spectrum(st.session_state.processed_data, st.session_state.samplerate))
        with tab2:
            st.pyplot(plot_waveform(st.session_state.processed_data, st.session_state.samplerate))
        with tab3:
            st.pyplot(plot_spectrogram(st.session_state.processed_data, st.session_state.samplerate))

else:
    st.info("Please upload an audio file to get started.")

