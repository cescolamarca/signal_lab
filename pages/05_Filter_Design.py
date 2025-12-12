import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from signallab.style import set_custom_style, get_plot_colors
from signallab.utils import format_number

# Apply custom style
set_custom_style(dark_mode=True)

st.markdown("# 🎚️ Filter Design")
st.markdown("Design and analyze digital filters (IIR/FIR) with live signal demonstration.")

# Expandable Theory Section
with st.expander("📚 Understanding Digital Filters"):
    st.markdown("""
    ### What are Digital Filters?
    
    Digital filters modify the frequency content of signals. They can:
    - **Remove noise** (lowpass to remove high-freq hiss)
    - **Isolate frequencies** (bandpass to extract specific tones)
    - **Remove interference** (notch/bandstop to eliminate power line hum)
    
    ### Filter Types
    
    | Type | Passes | Blocks | Common Use |
    |------|--------|--------|------------|
    | **Lowpass** | Low frequencies | High frequencies | Anti-aliasing, noise removal |
    | **Highpass** | High frequencies | Low frequencies | Removing DC offset, rumble |
    | **Bandpass** | A frequency band | Everything else | Isolating a signal of interest |
    | **Bandstop** | Everything except a band | A specific band | Removing 50/60 Hz hum |
    
    ### IIR vs FIR Filters
    
    **IIR (Infinite Impulse Response):**
    - Uses feedback → lower order needed
    - Non-linear phase → can cause distortion
    - Examples: Butterworth, Chebyshev, Elliptic
    
    **FIR (Finite Impulse Response):**
    - No feedback → always stable
    - Can have linear phase → no distortion
    - Higher order needed for sharp cutoff
    
    ### Filter Order
    
    Higher order = sharper transition between passband and stopband, but:
    - More computation
    - More delay (latency)
    - Potential ringing artifacts
    """)

# Sidebar Configuration
st.sidebar.header("🔧 Filter Specifications")

# Filter Class (IIR vs FIR)
filter_class = st.sidebar.radio("Filter Class", ["IIR", "FIR"])

if filter_class == "IIR":
    design_method = st.sidebar.selectbox(
        "Design Method", 
        ["Butterworth", "Chebyshev Type I", "Chebyshev Type II", "Elliptic", "Bessel"]
    )
else:
    design_method = st.sidebar.selectbox(
        "Window Type",
        ["Hamming", "Hanning", "Blackman", "Kaiser"]
    )

filter_type = st.sidebar.selectbox("Filter Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])

fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=1000.0, min_value=100.0)
nyquist = fs / 2.0

# Filter Order
if filter_class == "IIR":
    order = st.sidebar.slider("Filter Order", 1, 20, 4)
else:
    order = st.sidebar.slider("Number of Taps", 11, 201, 51, step=10)

# Dynamic Cutoff Inputs
st.sidebar.subheader("Cutoff Frequencies")
if filter_type in ["Lowpass", "Highpass"]:
    fc = st.sidebar.slider("Cutoff Frequency (Hz)", 10.0, nyquist - 10.0, nyquist / 4.0, step=5.0)
    wn = fc / nyquist
elif filter_type in ["Bandpass", "Bandstop"]:
    fc_range = st.sidebar.slider("Cutoff Frequencies (Hz)", 10.0, nyquist - 10.0, (nyquist/4.0, nyquist/2.0), step=5.0)
    wn = [f / nyquist for f in fc_range]

# Additional parameters for specific IIR filters
rp = None
rs = None
if filter_class == "IIR":
    if design_method == "Chebyshev Type I" or design_method == "Elliptic":
        rp = st.sidebar.slider("Passband Ripple (dB)", 0.1, 10.0, 1.0)
    if design_method == "Chebyshev Type II" or design_method == "Elliptic":
        rs = st.sidebar.slider("Stopband Attenuation (dB)", 10.0, 100.0, 40.0)
    if design_method == "Bessel" and filter_type in ["Bandpass", "Bandstop"]:
        st.sidebar.warning("Bessel filter may not work well with bandpass/bandstop.")

# FIR Kaiser beta
if filter_class == "FIR" and design_method == "Kaiser":
    kaiser_beta = st.sidebar.slider("Kaiser Beta", 0.0, 14.0, 5.0, step=0.5)

# Design Filter
try:
    if filter_class == "IIR":
        if design_method == "Butterworth":
            b, a = signal.butter(order, wn, btype=filter_type.lower())
        elif design_method == "Chebyshev Type I":
            b, a = signal.cheby1(order, rp, wn, btype=filter_type.lower())
        elif design_method == "Chebyshev Type II":
            b, a = signal.cheby2(order, rs, wn, btype=filter_type.lower())
        elif design_method == "Elliptic":
            b, a = signal.ellip(order, rp, rs, wn, btype=filter_type.lower())
        elif design_method == "Bessel":
            b, a = signal.bessel(order, wn, btype=filter_type.lower())
    else:  # FIR
        window_map = {
            "Hamming": "hamming",
            "Hanning": "hann", 
            "Blackman": "blackman",
            "Kaiser": ("kaiser", kaiser_beta) if design_method == "Kaiser" else "kaiser"
        }
        window = window_map.get(design_method, "hamming")
        
        if filter_type in ["Lowpass", "Highpass"]:
            b = signal.firwin(order, fc, fs=fs, window=window, pass_zero=(filter_type == "Lowpass"))
        else:
            b = signal.firwin(order, [fc_range[0], fc_range[1]], fs=fs, window=window, 
                             pass_zero=(filter_type == "Bandstop"))
        a = [1.0]
    
    # Frequency Response
    w, h = signal.freqz(b, a, worN=2000, fs=fs)
    
    colors = get_plot_colors()
    
    # === Section 1: Frequency Response ===
    st.subheader("📈 Frequency Response")
    
    fig_freq, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Magnitude
    mag_db = 20 * np.log10(np.abs(h) + 1e-10)
    ax_mag.plot(w, mag_db, color=colors[0], linewidth=2)
    ax_mag.set_title(f"{filter_class} {design_method} {filter_type} Filter - Order {order}")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_ylim(-80, 5)
    ax_mag.grid(True, alpha=0.3)
    
    # Mark cutoff frequencies
    ax_mag.axhline(-3, color='gray', linestyle='--', alpha=0.5, label='-3dB cutoff')
    if filter_type in ["Lowpass", "Highpass"]:
        ax_mag.axvline(fc, color=colors[1], linestyle='--', alpha=0.7, label=f'fc = {fc:.0f} Hz')
    else:
        ax_mag.axvline(fc_range[0], color=colors[1], linestyle='--', alpha=0.7, label=f'fc1 = {fc_range[0]:.0f} Hz')
        ax_mag.axvline(fc_range[1], color=colors[1], linestyle=':', alpha=0.7, label=f'fc2 = {fc_range[1]:.0f} Hz')
    ax_mag.legend(loc='upper right')

    # Phase
    angles = np.unwrap(np.angle(h))
    ax_phase.plot(w, np.degrees(angles), color=colors[2], linewidth=1)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_freq)
    
    # === Section 2: Pole-Zero Plot ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Pole-Zero Plot")
        z, p, k = signal.tf2zpk(b, a)
        
        fig_pz, ax_pz = plt.subplots(figsize=(5, 5))
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax_pz.plot(np.cos(theta), np.sin(theta), 'w--', alpha=0.3, linewidth=1)
        
        # Zeros and Poles
        ax_pz.scatter(np.real(z), np.imag(z), marker='o', s=80, facecolors='none', 
                     edgecolors=colors[2], linewidths=2, label=f'Zeros ({len(z)})')
        ax_pz.scatter(np.real(p), np.imag(p), marker='x', s=80, 
                     color=colors[1], linewidths=2, label=f'Poles ({len(p)})')
        
        ax_pz.set_xlim(-1.5, 1.5)
        ax_pz.set_ylim(-1.5, 1.5)
        ax_pz.set_aspect('equal')
        ax_pz.set_xlabel("Real")
        ax_pz.set_ylabel("Imaginary")
        ax_pz.legend(loc='upper right')
        ax_pz.grid(True, alpha=0.3)
        ax_pz.axhline(0, color='gray', linewidth=0.5)
        ax_pz.axvline(0, color='gray', linewidth=0.5)
        
        st.pyplot(fig_pz)
        
        # Stability check
        pole_magnitudes = np.abs(p)
        if len(pole_magnitudes) > 0 and np.all(pole_magnitudes < 1):
            st.success("✅ Filter is **stable** (all poles inside unit circle)")
        elif len(pole_magnitudes) > 0:
            st.error("⚠️ Filter is **unstable** (poles outside unit circle)")
    
    with col2:
        st.subheader("📉 Impulse Response")
        try:
            imp_length = min(200, int(fs / 2))
            t_imp, y_imp = signal.dimpulse((b, a, 1/fs), n=imp_length)
            y_imp = y_imp[0].flatten()
            
            fig_imp, ax_imp = plt.subplots(figsize=(5, 5))
            ax_imp.stem(t_imp[:len(y_imp)], y_imp, linefmt=colors[3], markerfmt='o', basefmt=" ")
            ax_imp.set_xlabel("Time (s)")
            ax_imp.set_ylabel("Amplitude")
            ax_imp.grid(True, alpha=0.3)
            st.pyplot(fig_imp)
        except Exception:
            st.warning("Could not compute impulse response for this configuration.")
    
    # === Section 3: Live Signal Demo ===
    st.markdown("---")
    st.subheader("🔬 Live Filter Demo")
    st.markdown("See how the filter affects a real signal with noise!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        demo_signal_type = st.selectbox("Demo Signal", [
            "Sine with High-Freq Noise",
            "Sine with Low-Freq Drift", 
            "Two Tones + Noise",
            "Chirp Signal"
        ])
    
    with col2:
        demo_noise_level = st.slider("Noise Level", 0.0, 1.0, 0.3, step=0.05)
    
    # Generate demo signal
    demo_duration = 0.5
    demo_t = np.linspace(0, demo_duration, int(demo_duration * fs), endpoint=False)
    
    if demo_signal_type == "Sine with High-Freq Noise":
        demo_clean = np.sin(2 * np.pi * 50 * demo_t)  # 50 Hz tone
        demo_noise = demo_noise_level * np.sin(2 * np.pi * 400 * demo_t)  # 400 Hz noise
        expected_filter = "Lowpass < 200 Hz"
    elif demo_signal_type == "Sine with Low-Freq Drift":
        demo_clean = np.sin(2 * np.pi * 200 * demo_t)  # 200 Hz tone
        demo_noise = demo_noise_level * np.sin(2 * np.pi * 10 * demo_t)  # 10 Hz drift
        expected_filter = "Highpass > 50 Hz"
    elif demo_signal_type == "Two Tones + Noise":
        demo_clean = np.sin(2 * np.pi * 100 * demo_t) + 0.5 * np.sin(2 * np.pi * 150 * demo_t)
        demo_noise = demo_noise_level * np.random.randn(len(demo_t))
        expected_filter = "Bandpass 50-200 Hz"
    else:  # Chirp
        demo_clean = signal.chirp(demo_t, 50, demo_duration, 300)
        demo_noise = demo_noise_level * np.random.randn(len(demo_t))
        expected_filter = "Lowpass to smooth"
    
    demo_signal = demo_clean + demo_noise
    
    # Apply filter
    demo_filtered = signal.filtfilt(b, a, demo_signal)
    
    # Plot comparison
    fig_demo, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    # Time domain - Original
    axes[0, 0].plot(demo_t * 1000, demo_signal, color=colors[0], linewidth=0.8, alpha=0.8)
    axes[0, 0].plot(demo_t * 1000, demo_clean, color='white', linewidth=1, alpha=0.3, linestyle='--', label='Clean')
    axes[0, 0].set_title("Original (Noisy)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time domain - Filtered
    axes[0, 1].plot(demo_t * 1000, demo_filtered, color=colors[3], linewidth=0.8)
    axes[0, 1].plot(demo_t * 1000, demo_clean, color='white', linewidth=1, alpha=0.3, linestyle='--', label='Clean')
    axes[0, 1].set_title("Filtered")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectrum - Original
    f_orig, Pxx_orig = signal.welch(demo_signal, fs, nperseg=256)
    axes[1, 0].semilogy(f_orig, Pxx_orig, color=colors[0])
    axes[1, 0].set_title("Original Spectrum")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spectrum - Filtered
    f_filt, Pxx_filt = signal.welch(demo_filtered, fs, nperseg=256)
    axes[1, 1].semilogy(f_filt, Pxx_filt, color=colors[3])
    axes[1, 1].set_title("Filtered Spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_demo)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    mse_before = np.mean((demo_signal - demo_clean) ** 2)
    mse_after = np.mean((demo_filtered - demo_clean) ** 2)
    improvement = (1 - mse_after / mse_before) * 100 if mse_before > 0 else 0
    
    col1.metric("MSE (Before)", f"{mse_before:.4f}")
    col2.metric("MSE (After)", f"{mse_after:.4f}")
    col3.metric("Noise Reduction", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    if improvement < 20:
        st.info(f"💡 **Tip:** For this signal, try a **{expected_filter}** filter for better results!")

except Exception as e:
    st.error(f"Error designing filter: {e}")
    st.info("Try adjusting the filter parameters. Some combinations may not be valid.")
