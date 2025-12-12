import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import io
import base64
from signallab.style import set_custom_style, get_plot_colors

# Apply custom style
set_custom_style(dark_mode=True)

st.markdown("# 🎵 Audio Noise Removal")
st.markdown("Demonstrate noise reduction using digital filters on audio signals.")

# Sidebar Configuration
st.sidebar.header("🎛️ Audio Configuration")

# Audio Source Selection
audio_source = st.sidebar.radio(
    "Audio Source",
    ["Generate Synthetic Audio", "Upload WAV File"]
)

# Parameters for synthetic audio
if audio_source == "Generate Synthetic Audio":
    st.sidebar.subheader("Signal Parameters")
    base_freq = st.sidebar.slider("Base Frequency (Hz)", 100, 1000, 440, step=10)
    duration = st.sidebar.slider("Duration (s)", 0.5, 5.0, 2.0, step=0.5)
    sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [8000, 16000, 22050, 44100], index=2)
    
    st.sidebar.subheader("Noise Configuration")
    noise_type = st.sidebar.selectbox("Noise Type", ["White Noise", "High-Frequency Hum", "Power Line (50/60 Hz)"])
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, step=0.05)
    
    if noise_type == "High-Frequency Hum":
        hum_freq = st.sidebar.slider("Hum Frequency (Hz)", 1000, 5000, 3000)
    elif noise_type == "Power Line (50/60 Hz)":
        power_freq = st.sidebar.radio("Power Line Freq", [50, 60])

# Filter Configuration
st.sidebar.header("🔧 Filter Design")
filter_class = st.sidebar.selectbox("Filter Class", ["IIR (Butterworth)", "IIR (Chebyshev I)", "FIR (Window Method)"])
filter_type = st.sidebar.selectbox("Filter Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])

# Helper function to create audio player HTML
def get_audio_player(audio_data, sample_rate):
    """Generate HTML audio player from numpy array."""
    # Normalize to 16-bit range
    audio_normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    
    # Write to bytes buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_normalized)
    buffer.seek(0)
    
    # Encode to base64
    audio_base64 = base64.b64encode(buffer.read()).decode()
    
    return f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'

# Generate or load audio
audio_data = None
fs = 22050  # Default sample rate

if audio_source == "Generate Synthetic Audio":
    fs = sample_rate
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Generate clean signal (musical tone with harmonics)
    clean_signal = np.sin(2 * np.pi * base_freq * t)
    clean_signal += 0.5 * np.sin(2 * np.pi * 2 * base_freq * t)  # 2nd harmonic
    clean_signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)  # 3rd harmonic
    
    # Apply amplitude envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * fs)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    clean_signal *= envelope
    
    # Generate noise
    if noise_type == "White Noise":
        noise = noise_level * np.random.randn(len(t))
    elif noise_type == "High-Frequency Hum":
        noise = noise_level * np.sin(2 * np.pi * hum_freq * t)
        noise += noise_level * 0.5 * np.sin(2 * np.pi * hum_freq * 1.5 * t)
    elif noise_type == "Power Line (50/60 Hz)":
        noise = noise_level * np.sin(2 * np.pi * power_freq * t)
        noise += noise_level * 0.3 * np.sin(2 * np.pi * power_freq * 3 * t)  # 3rd harmonic
    
    # Combine
    audio_data = clean_signal + noise
    original_clean = clean_signal
    
else:  # Upload WAV file
    uploaded_file = st.sidebar.file_uploader("Upload WAV File", type=['wav'])
    if uploaded_file is not None:
        fs, audio_data = wavfile.read(uploaded_file)
        # Convert to float and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(float) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(float) / 2147483648.0
        # Handle stereo by taking first channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        original_clean = None  # No clean reference for uploaded files
    else:
        st.info("👆 Please upload a WAV file or switch to synthetic audio generation.")
        st.stop()

# Nyquist frequency
nyquist = fs / 2.0

# Dynamic cutoff based on filter type
st.sidebar.subheader("Cutoff Frequencies")
if filter_type in ["Lowpass", "Highpass"]:
    cutoff = st.sidebar.slider("Cutoff (Hz)", 50, int(nyquist - 50), int(nyquist / 4))
    wn = cutoff / nyquist
else:
    cutoff_low = st.sidebar.slider("Low Cutoff (Hz)", 50, int(nyquist / 2), 200)
    cutoff_high = st.sidebar.slider("High Cutoff (Hz)", int(cutoff_low + 50), int(nyquist - 50), min(int(nyquist / 2), cutoff_low + 500))
    wn = [cutoff_low / nyquist, cutoff_high / nyquist]

order = st.sidebar.slider("Filter Order", 2, 20, 6)

# Design and apply filter
try:
    if filter_class == "IIR (Butterworth)":
        b, a = signal.butter(order, wn, btype=filter_type.lower())
    elif filter_class == "IIR (Chebyshev I)":
        b, a = signal.cheby1(order, 1, wn, btype=filter_type.lower())
    elif filter_class == "FIR (Window Method)":
        numtaps = order * 10 + 1  # FIR needs more taps
        if filter_type in ["Lowpass", "Highpass"]:
            b = signal.firwin(numtaps, cutoff, fs=fs, pass_zero=(filter_type == "Lowpass"))
        else:
            b = signal.firwin(numtaps, [cutoff_low, cutoff_high], fs=fs, pass_zero=(filter_type == "Bandstop"))
        a = [1.0]  # FIR has no feedback
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    # ======= VISUALIZATION =======
    colors = get_plot_colors()
    
    # === Section 1: Original Audio ===
    st.markdown("## 📊 Original Signal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Waveform")
        fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
        t_plot = np.linspace(0, len(audio_data) / fs, len(audio_data))
        ax_wave.plot(t_plot, audio_data, color=colors[0], linewidth=0.5)
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title("Original (Noisy) Signal")
        st.pyplot(fig_wave)
    
    with col2:
        st.markdown("### Spectrogram")
        fig_spec, ax_spec = plt.subplots(figsize=(6, 3))
        Pxx, freqs, bins, im = ax_spec.specgram(audio_data, NFFT=512, Fs=fs, noverlap=256, cmap='inferno')
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title("Original Spectrogram")
        ax_spec.set_ylim(0, min(nyquist, 5000))  # Cap at 5kHz for visibility
        fig_spec.colorbar(im, ax=ax_spec, label='dB')
        st.pyplot(fig_spec)
    
    # Audio Player for Original
    st.markdown("**🔊 Listen to Original:**")
    st.markdown(get_audio_player(audio_data, fs), unsafe_allow_html=True)
    
    # === Section 2: Frequency Analysis ===
    st.markdown("## 🔍 Frequency Analysis (FFT)")
    
    # Compute FFT
    N = len(audio_data)
    yf = np.fft.rfft(audio_data)
    xf = np.fft.rfftfreq(N, 1/fs)
    magnitude = np.abs(yf) / N
    
    # Find peaks
    peak_indices = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)[0]
    peak_freqs = xf[peak_indices]
    peak_mags = magnitude[peak_indices]
    
    fig_fft, ax_fft = plt.subplots(figsize=(10, 4))
    ax_fft.plot(xf, magnitude, color=colors[0], linewidth=1)
    ax_fft.scatter(peak_freqs, peak_mags, color=colors[1], s=50, zorder=5, label='Peaks')
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_title("Frequency Spectrum with Peak Detection")
    ax_fft.set_xlim(0, min(nyquist, 5000))
    ax_fft.legend()
    ax_fft.grid(True, alpha=0.3)
    st.pyplot(fig_fft)
    
    # Show detected frequencies
    if len(peak_freqs) > 0:
        st.markdown("**Detected Frequency Components:**")
        freq_display = ", ".join([f"{f:.1f} Hz" for f in sorted(peak_freqs[:10])])
        st.info(f"🎵 {freq_display}")
    
    # === Section 3: Filter Response ===
    st.markdown("## 🎚️ Filter Frequency Response")
    
    w, h = signal.freqz(b, a, worN=2000, fs=fs)
    
    fig_resp, ax_resp = plt.subplots(figsize=(10, 4))
    ax_resp.plot(w, 20 * np.log10(np.abs(h) + 1e-10), color=colors[2], linewidth=2)
    ax_resp.set_xlabel("Frequency (Hz)")
    ax_resp.set_ylabel("Magnitude (dB)")
    ax_resp.set_title(f"{filter_class} {filter_type} Filter Response")
    ax_resp.set_xlim(0, min(nyquist, 5000))
    ax_resp.set_ylim(-80, 5)
    ax_resp.axhline(-3, color='gray', linestyle='--', alpha=0.5, label='-3dB')
    if filter_type in ["Lowpass", "Highpass"]:
        ax_resp.axvline(cutoff, color=colors[1], linestyle='--', alpha=0.7, label=f'Cutoff: {cutoff}Hz')
    ax_resp.legend()
    ax_resp.grid(True, alpha=0.3)
    st.pyplot(fig_resp)
    
    # === Section 4: Filtered Result ===
    st.markdown("## ✨ Filtered Signal (Noise Removed)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Waveform")
        fig_filt, ax_filt = plt.subplots(figsize=(6, 3))
        ax_filt.plot(t_plot, filtered_audio, color=colors[3], linewidth=0.5)
        ax_filt.set_xlabel("Time (s)")
        ax_filt.set_ylabel("Amplitude")
        ax_filt.set_title("Filtered Signal")
        st.pyplot(fig_filt)
    
    with col2:
        st.markdown("### Spectrogram")
        fig_spec2, ax_spec2 = plt.subplots(figsize=(6, 3))
        Pxx2, freqs2, bins2, im2 = ax_spec2.specgram(filtered_audio, NFFT=512, Fs=fs, noverlap=256, cmap='inferno')
        ax_spec2.set_xlabel("Time (s)")
        ax_spec2.set_ylabel("Frequency (Hz)")
        ax_spec2.set_title("Filtered Spectrogram")
        ax_spec2.set_ylim(0, min(nyquist, 5000))
        fig_spec2.colorbar(im2, ax=ax_spec2, label='dB')
        st.pyplot(fig_spec2)
    
    # Audio Player for Filtered
    st.markdown("**🔊 Listen to Filtered:**")
    st.markdown(get_audio_player(filtered_audio, fs), unsafe_allow_html=True)
    
    # === Section 5: Comparison Metrics ===
    st.markdown("## 📈 Noise Reduction Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics
    original_power = np.mean(audio_data ** 2)
    filtered_power = np.mean(filtered_audio ** 2)
    
    if original_clean is not None:
        # If we have clean reference, calculate actual SNR improvement
        noise_orig = audio_data - original_clean
        noise_filt = filtered_audio - original_clean
        snr_orig = 10 * np.log10(np.mean(original_clean**2) / (np.mean(noise_orig**2) + 1e-10))
        snr_filt = 10 * np.log10(np.mean(original_clean**2) / (np.mean(noise_filt**2) + 1e-10))
        snr_improvement = snr_filt - snr_orig
        col1.metric("Original SNR", f"{snr_orig:.1f} dB")
        col2.metric("Filtered SNR", f"{snr_filt:.1f} dB")
        col3.metric("SNR Improvement", f"+{snr_improvement:.1f} dB", delta=f"{snr_improvement:.1f} dB")
    else:
        # Estimate based on power reduction in expected noise band
        power_reduction = 10 * np.log10(filtered_power / (original_power + 1e-10))
        col1.metric("Original RMS", f"{np.sqrt(original_power):.4f}")
        col2.metric("Filtered RMS", f"{np.sqrt(filtered_power):.4f}")
        col3.metric("Power Change", f"{power_reduction:.1f} dB")
    
    # === Theory Section ===
    with st.expander("📚 How Digital Noise Filtering Works"):
        st.markdown("""
        ### The Noise Removal Process
        
        1. **Analyze the Signal**: Use FFT to identify which frequencies contain the desired signal vs noise.
        
        2. **Design a Filter**: Create a filter that passes desired frequencies and attenuates noise:
           - **Lowpass**: Removes high-frequency noise (hiss, hum)
           - **Highpass**: Removes low-frequency noise (rumble, DC offset)
           - **Bandpass**: Isolates a specific frequency range
           - **Bandstop (Notch)**: Removes a specific frequency (power line hum)
        
        3. **Apply the Filter**: The filter is convolved with the signal, attenuating unwanted frequencies.
        
        ### IIR vs FIR Filters
        
        | Feature | IIR (Butterworth, Chebyshev) | FIR (Window Method) |
        |---------|------------------------------|---------------------|
        | Order needed | Lower (efficient) | Higher (more taps) |
        | Phase response | Non-linear | Can be linear |
        | Stability | Can be unstable | Always stable |
        | Use case | Real-time, low latency | Precision, post-processing |
        
        ### Key Metrics
        
        - **SNR (Signal-to-Noise Ratio)**: Higher is better. Measures how much louder the signal is than noise.
        - **Filter Order**: Higher order = sharper cutoff, but more computational cost and potential ringing.
        """)

except Exception as e:
    st.error(f"Error processing audio: {e}")
    st.info("Try adjusting filter parameters or check your audio file format.")
