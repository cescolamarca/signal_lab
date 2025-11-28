import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from signallab.style import set_custom_style

# Apply custom style
set_custom_style(dark_mode=True)

st.title("Frequency Domain Analysis 📊")
st.markdown("Analyze the frequency content of signals using the Fast Fourier Transform (FFT).")

# --- Sidebar Controls ---
st.sidebar.header("Signal Generation")

fs = st.sidebar.slider("Sampling Frequency (Hz)", 100, 2000, 1000, 100)
duration = st.sidebar.slider("Duration (s)", 0.1, 5.0, 1.0, 0.1)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Signal Components
st.sidebar.subheader("Signal Components")
num_components = st.sidebar.number_input("Number of Sinusoids", 1, 5, 2)

signal = np.zeros_like(t)
components_info = []

for i in range(num_components):
    st.sidebar.markdown(f"**Component {i+1}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        freq = st.number_input(f"Freq {i+1} (Hz)", 0.0, fs/2.0, 10.0 * (i+1), key=f"f{i}")
    with col2:
        amp = st.number_input(f"Amp {i+1}", 0.0, 10.0, 1.0, key=f"a{i}")
    
    signal += amp * np.cos(2 * np.pi * freq * t)
    components_info.append(f"{amp}*cos(2π*{freq}*t)")

# Noise
add_noise = st.sidebar.checkbox("Add Gaussian Noise")
if add_noise:
    noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5)
    noise = np.random.normal(0, noise_level, len(t))
    signal += noise

# Windowing
st.sidebar.subheader("Windowing")
window_type = st.sidebar.selectbox("Window Function", ["Rectangular", "Hamming", "Hanning", "Blackman"])

if window_type == "Hamming":
    window = np.hamming(len(signal))
elif window_type == "Hanning":
    window = np.hanning(len(signal))
elif window_type == "Blackman":
    window = np.blackman(len(signal))
else:
    window = np.ones_like(signal)

windowed_signal = signal * window

# --- Visualization ---

# 1. Time Domain Plot
st.subheader("1. Time Domain Signal")
fig_time, ax_time = plt.subplots(figsize=(10, 4))
ax_time.plot(t[:500], windowed_signal[:500], label="Signal (First 500 samples)") # Limit points for performance/clarity
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Amplitude")
ax_time.set_title("Time Domain Signal x(t)")
ax_time.grid(True, alpha=0.3)
ax_time.legend()
st.pyplot(fig_time)

# 2. Frequency Domain (FFT)
st.subheader("2. Frequency Domain (Magnitude Spectrum)")

# Compute FFT
N = len(signal)
yf = fft(windowed_signal)
xf = fftfreq(N, 1/fs)

# Shift zero frequency to center (optional, usually 0 to fs/2 is enough for real signals)
# But standard FFT plot for real signals is usually 0 to fs/2
xf_positive = xf[:N//2]
yf_magnitude = 2.0/N * np.abs(yf[0:N//2])

fig_freq, ax_freq = plt.subplots(figsize=(10, 4))
ax_freq.plot(xf_positive, yf_magnitude, color='C1')
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Magnitude")
ax_freq.set_title("Magnitude Spectrum |X(f)|")
ax_freq.grid(True, alpha=0.3)
st.pyplot(fig_freq)

# 3. Spectrogram (Optional but cool)
st.subheader("3. Spectrogram")
fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
Pxx, freqs, bins, im = ax_spec.specgram(windowed_signal, NFFT=256, Fs=fs, noverlap=128, cmap='inferno')
ax_spec.set_xlabel("Time (s)")
ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_title("Spectrogram")
fig_spec.colorbar(im, ax=ax_spec, label='Intensity (dB)')
st.pyplot(fig_spec)

st.markdown("---")
st.markdown("**Theory Note:**")
st.latex(r"X(k) = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}")
