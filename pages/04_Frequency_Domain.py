import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from signallab.style import set_custom_style, get_plot_colors
from signallab.utils import parse_expression, format_number

# Apply custom style
set_custom_style(dark_mode=True)

st.markdown("# 📊 Frequency Domain Analysis")
st.markdown("Analyze signals in the frequency domain using the Fast Fourier Transform (FFT).")

# Expandable Theory Section
with st.expander("📚 Understanding FFT and Spectral Analysis"):
    st.markdown("""
    ### The Fourier Transform
    
    The **Fast Fourier Transform (FFT)** decomposes a signal into its constituent frequencies. 
    Any signal can be represented as a sum of sinusoids at different frequencies.
    
    **Key Concepts:**
    - **Magnitude Spectrum**: Shows the amplitude of each frequency component
    - **Phase Spectrum**: Shows the phase offset of each frequency component
    - **Spectrogram**: Shows how the frequency content changes over time
    
    ### Windowing Functions
    
    Windowing reduces **spectral leakage** - artifacts caused by analyzing finite-length signals:
    
    | Window | Main lobe width | Side lobe level | Use case |
    |--------|-----------------|-----------------|----------|
    | None (Rectangular) | Narrowest | Highest (-13 dB) | When signal fits exactly in window |
    | Hamming | Medium | Low (-43 dB) | General purpose |
    | Hanning | Medium | Low (-32 dB) | General purpose |
    | Blackman | Widest | Lowest (-58 dB) | When side lobe suppression critical |
    
    ### Frequency Resolution
    
    Resolution = Sample Rate / Number of Samples = fs / N
    
    More samples = finer frequency resolution, but requires longer signal duration.
    """)

# Sidebar controls
st.sidebar.header("📡 Signal Configuration")

# Signal Source Selection
signal_source = st.sidebar.radio(
    "Signal Source",
    ["Preset Signals", "Custom Expression"]
)

# Preset signals for demo-worthy displays
preset_signals = {
    "Clean Dual Tone": {
        "expr": "sin(2*pi*100*t) + 0.6*sin(2*pi*250*t)",
        "desc": "Two pure tones at 100 Hz and 250 Hz"
    },
    "Noisy Sine Wave": {
        "expr": "sin(2*pi*50*t) + 0.3*np.random.randn(len(t))",
        "desc": "50 Hz sine wave buried in white noise"
    },
    "AM Modulated Signal": {
        "expr": "(1 + 0.5*sin(2*pi*10*t)) * sin(2*pi*200*t)",
        "desc": "Amplitude modulated carrier with sidebands"
    },
    "Frequency Chirp": {
        "expr": "sin(2*pi*(20 + 80*t/duration)*t)",
        "desc": "Swept frequency from 20 Hz to 100 Hz"
    },
    "Square Wave (with harmonics)": {
        "expr": "np.sign(sin(2*pi*25*t))",
        "desc": "Square wave showing odd harmonics"
    },
    "Multi-Component Signal": {
        "expr": "sin(2*pi*30*t) + 0.5*sin(2*pi*75*t) + 0.3*sin(2*pi*120*t) + 0.2*sin(2*pi*180*t)",
        "desc": "Complex signal with 4 frequency components"
    }
}

if signal_source == "Preset Signals":
    preset_name = st.sidebar.selectbox("Choose Preset", list(preset_signals.keys()))
    expression = preset_signals[preset_name]["expr"]
    st.sidebar.info(preset_signals[preset_name]["desc"])
else:
    default_expr = "sin(2*pi*50*t) + 0.5*sin(2*pi*120*t)"
    expression = st.sidebar.text_area("Expression (use 't')", value=default_expr, height=100)

# Time parameters
duration = st.sidebar.slider("Duration (s)", 0.1, 5.0, 1.0, step=0.1)
fs = st.sidebar.slider("Sampling Rate (Hz)", 100, 4000, 1000, step=100)

# Windowing
window_type = st.sidebar.selectbox("Window Function", ["None", "Hamming", "Hanning", "Blackman", "Bartlett"])

# FFT Parameters
st.sidebar.subheader("FFT Settings")
show_negative = st.sidebar.checkbox("Show Negative Frequencies", value=False)
log_scale = st.sidebar.checkbox("Logarithmic Scale (dB)", value=False)

# Spectrogram settings
st.sidebar.subheader("Spectrogram Settings")
nfft = st.sidebar.select_slider("NFFT", options=[64, 128, 256, 512, 1024], value=256)
overlap_pct = st.sidebar.slider("Overlap %", 0, 90, 50, step=10)

# Time vector
t = np.linspace(0, duration, int(duration * fs), endpoint=False)

try:
    # Generate Signal - handle special expressions that need context
    context = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "t": t,
        "np": np,
        "duration": duration
    }
    y = eval(expression, {"__builtins__": None}, context)
    
    # Ensure y is an array
    if np.isscalar(y):
        y = np.full_like(t, y)
    
    # Apply Window
    if window_type != "None":
        window_funcs = {
            "Hamming": np.hamming,
            "Hanning": np.hanning,
            "Blackman": np.blackman,
            "Bartlett": np.bartlett
        }
        window = window_funcs[window_type](len(y))
        y_windowed = y * window
    else:
        y_windowed = y

    # Perform FFT
    N = len(y)
    yf = np.fft.fft(y_windowed)
    xf = np.fft.fftfreq(N, 1/fs)
    
    colors = get_plot_colors()
    
    # === Section 1: Time Domain ===
    st.subheader("🕐 Time Domain")
    fig_time, ax_time = plt.subplots(figsize=(10, 3))
    ax_time.plot(t, y, color=colors[0], label="Original Signal", linewidth=1)
    if window_type != "None":
        ax_time.plot(t, y_windowed, color=colors[1], alpha=0.7, linestyle='--', label=f"{window_type} Windowed")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    st.pyplot(fig_time)
    
    # === Section 2: Frequency Domain ===
    st.subheader("📈 Frequency Domain")
    
    tab1, tab2, tab3 = st.tabs(["Magnitude Spectrum", "Phase Spectrum", "Spectrogram"])
    
    with tab1:
        if show_negative:
            # Centered spectrum
            yf_shifted = np.fft.fftshift(yf)
            xf_shifted = np.fft.fftshift(xf)
            magnitude = np.abs(yf_shifted) / N
            freq_axis = xf_shifted
        else:
            # Positive frequencies only (more common view)
            positive_mask = xf >= 0
            magnitude = 2 * np.abs(yf[positive_mask]) / N  # 2x for single-sided
            magnitude[0] /= 2  # DC component not doubled
            freq_axis = xf[positive_mask]
        
        # Apply log scale if selected
        if log_scale:
            magnitude_plot = 20 * np.log10(magnitude + 1e-10)
            ylabel = "Magnitude (dB)"
        else:
            magnitude_plot = magnitude
            ylabel = "Magnitude"
        
        fig_mag, ax_mag = plt.subplots(figsize=(10, 4))
        ax_mag.plot(freq_axis, magnitude_plot, color=colors[2], linewidth=1)
        
        # Peak Detection
        peak_threshold = np.max(magnitude) * 0.1
        if not log_scale:
            peak_indices = signal.find_peaks(magnitude, height=peak_threshold, distance=5)[0]
            peak_freqs = freq_axis[peak_indices] if not show_negative else freq_axis[peak_indices]
            peak_mags = magnitude_plot[peak_indices]
            
            # Only show positive peaks for clarity
            if not show_negative:
                ax_mag.scatter(peak_freqs, peak_mags, color=colors[1], s=60, zorder=5, marker='v', label='Peaks')
                
                # Annotate peaks
                for f, m in zip(peak_freqs[:5], peak_mags[:5]):  # Limit to top 5
                    if f > 0:
                        ax_mag.annotate(f'{f:.1f} Hz', (f, m), textcoords="offset points", 
                                       xytext=(0, 10), ha='center', fontsize=8, color=colors[1])
        
        ax_mag.set_xlabel("Frequency (Hz)")
        ax_mag.set_ylabel(ylabel)
        ax_mag.set_title("Magnitude Spectrum with Peak Detection")
        ax_mag.legend()
        ax_mag.grid(True, alpha=0.3)
        st.pyplot(fig_mag)
        
        # Show detected frequencies
        if not log_scale and len(peak_freqs) > 0:
            positive_peaks = [f for f in peak_freqs if f > 1]  # Filter out DC
            if positive_peaks:
                st.success(f"🎵 **Detected Frequencies:** {', '.join([f'{f:.1f} Hz' for f in sorted(positive_peaks)[:8]])}")
        
    with tab2:
        if show_negative:
            phase = np.angle(yf_shifted)
            freq_for_phase = xf_shifted
        else:
            phase = np.angle(yf[positive_mask])
            freq_for_phase = freq_axis
        
        fig_phase, ax_phase = plt.subplots(figsize=(10, 4))
        ax_phase.plot(freq_for_phase, phase, color=colors[3], linewidth=0.5)
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (radians)")
        ax_phase.set_title("Phase Spectrum")
        ax_phase.grid(True, alpha=0.3)
        st.pyplot(fig_phase)

    with tab3:
        noverlap = int(nfft * overlap_pct / 100)
        fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
        Pxx, freqs, bins, im = ax_spec.specgram(y_windowed, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap='inferno')
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title(f"Spectrogram (NFFT={nfft}, Overlap={overlap_pct}%)")
        fig_spec.colorbar(im, ax=ax_spec, label='Power/Frequency (dB/Hz)')
        st.pyplot(fig_spec)
        
        st.info("💡 **Tip:** Increase NFFT for better frequency resolution, decrease for better time resolution.")

    # === Section 3: Signal Properties ===
    st.markdown("### 📊 Signal Properties")
    col1, col2, col3, col4 = st.columns(4)
    
    # DC Component
    dc = np.abs(yf[0]) / N
    col1.metric("DC Component", format_number(dc, 3))
    
    # Energy
    energy = np.sum(y**2) / len(y)
    col2.metric("Signal Power", format_number(energy, 3))
    
    # Dominant Frequency
    positive_yf = np.abs(yf[:N//2])
    dominant_idx = np.argmax(positive_yf[1:]) + 1  # Skip DC
    dominant_freq = xf[dominant_idx]
    col3.metric("Dominant Freq", f"{abs(dominant_freq):.1f} Hz")
    
    # Bandwidth estimate (where 90% energy)
    sorted_mags = np.sort(np.abs(yf[:N//2]))[::-1]
    cumsum = np.cumsum(sorted_mags**2)
    bw_idx = np.searchsorted(cumsum, 0.9 * cumsum[-1])
    col4.metric("Freq Resolution", f"{fs/N:.2f} Hz")

except Exception as e:
    st.error(f"Error processing signal: {e}")
    st.info("Ensure your expression is valid. Supported functions: sin, cos, exp, sqrt, abs, np.random.randn(len(t)).")
