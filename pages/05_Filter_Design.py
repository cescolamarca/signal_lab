import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from signallab.style import set_custom_style

# Apply custom style
set_custom_style(dark_mode=True)

st.title("Digital Filter Design 🛠️")
st.markdown("Design and analyze Infinite Impulse Response (IIR) and Finite Impulse Response (FIR) filters.")

# --- Sidebar Controls ---
st.sidebar.header("Filter Specifications")

filter_type = st.sidebar.selectbox("Filter Response Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])
design_method = st.sidebar.selectbox("Design Method", ["IIR - Butterworth", "IIR - Chebyshev Type I", "IIR - Chebyshev Type II", "IIR - Elliptic", "FIR - Window Method"])

order = st.sidebar.slider("Filter Order", 1, 20, 4)
fs = st.sidebar.number_input("Sampling Frequency (Hz)", 100.0, 100000.0, 1000.0, 100.0)
nyquist = 0.5 * fs

# Cutoff Frequencies
if filter_type in ["Bandpass", "Bandstop"]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fc1 = st.number_input("Cutoff 1 (Hz)", 0.0, nyquist, nyquist * 0.2)
    with col2:
        fc2 = st.number_input("Cutoff 2 (Hz)", 0.0, nyquist, nyquist * 0.4)
    cutoff = [fc1, fc2]
    # Validation
    if fc1 >= fc2:
        st.error("Cutoff 1 must be less than Cutoff 2")
        st.stop()
else:
    fc = st.sidebar.number_input("Cutoff Frequency (Hz)", 0.0, nyquist, nyquist * 0.25)
    cutoff = fc

# Additional Parameters for specific filters
rp = 0
rs = 0
if "Chebyshev Type I" in design_method or "Elliptic" in design_method:
    rp = st.sidebar.number_input("Passband Ripple (dB)", 0.1, 10.0, 1.0)
if "Chebyshev Type II" in design_method or "Elliptic" in design_method:
    rs = st.sidebar.number_input("Stopband Attenuation (dB)", 10.0, 100.0, 40.0)

# Window selection for FIR
window_name = "hamming"
if "FIR" in design_method:
    window_name = st.sidebar.selectbox("Window Function", ["rectangular", "hamming", "hanning", "blackman"])

# --- Filter Design ---
b, a = None, None

try:
    if "Butterworth" in design_method:
        b, a = signal.butter(order, cutoff, btype=filter_type.lower(), fs=fs)
    elif "Chebyshev Type I" in design_method:
        b, a = signal.cheby1(order, rp, cutoff, btype=filter_type.lower(), fs=fs)
    elif "Chebyshev Type II" in design_method:
        b, a = signal.cheby2(order, rs, cutoff, btype=filter_type.lower(), fs=fs)
    elif "Elliptic" in design_method:
        b, a = signal.ellip(order, rp, rs, cutoff, btype=filter_type.lower(), fs=fs)
    elif "FIR" in design_method:
        # FIR Design using firwin
        # firwin requires cutoff to be normalized if fs is not provided, but we can pass fs
        # numtaps must be odd for highpass/bandstop with antisymmetric linear phase, but firwin handles some cases.
        # Let's use order + 1 taps
        numtaps = order + 1
        
        # Adjust pass_zero for Bandpass/Bandstop
        pass_zero = True
        if filter_type == "Lowpass": pass_zero = True
        elif filter_type == "Highpass": pass_zero = False
        elif filter_type == "Bandpass": pass_zero = False
        elif filter_type == "Bandstop": pass_zero = True

        # firwin uses 'boxcar' for rectangular
        win = 'boxcar' if window_name == 'rectangular' else window_name
        
        b = signal.firwin(numtaps, cutoff, window=win, fs=fs, pass_zero=pass_zero)
        a = [1.0]

except Exception as e:
    st.error(f"Error designing filter: {e}")
    st.stop()

# --- Visualization ---

# 1. Frequency Response
st.subheader("1. Frequency Response")
w, h_freq = signal.freqz(b, a, fs=fs)

fig_freq, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Magnitude
ax_mag.plot(w, 20 * np.log10(abs(h_freq)), 'C0')
ax_mag.set_title("Magnitude Response (Bode Plot)")
ax_mag.set_ylabel("Amplitude (dB)")
ax_mag.grid(True, which='both', alpha=0.3)
ax_mag.axvline(nyquist, color='r', linestyle='--', alpha=0.5, label='Nyquist')

# Phase
angles = np.unwrap(np.angle(h_freq))
ax_phase.plot(w, angles, 'C1')
ax_phase.set_title("Phase Response")
ax_phase.set_xlabel("Frequency (Hz)")
ax_phase.set_ylabel("Phase (radians)")
ax_phase.grid(True, which='both', alpha=0.3)

st.pyplot(fig_freq)

col1, col2 = st.columns(2)

# 2. Impulse Response
with col1:
    st.subheader("2. Impulse Response")
    # Generate impulse
    impulse = np.zeros(100)
    impulse[0] = 1
    if len(a) == 1: # FIR
        h_imp = b
        t_imp = np.arange(len(b)) / fs
    else: # IIR
        t_imp, h_imp = signal.dimpulse((b, a, 1/fs), n=100)
        t_imp = np.array(t_imp).flatten()
        h_imp = np.array(h_imp).flatten()

    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
    ax_imp.stem(t_imp, h_imp, basefmt=" ")
    ax_imp.set_title("Impulse Response h[n]")
    ax_imp.set_xlabel("Time (s)")
    ax_imp.grid(True, alpha=0.3)
    st.pyplot(fig_imp)

# 3. Pole-Zero Plot
with col2:
    st.subheader("3. Pole-Zero Plot")
    z, p, k = signal.tf2zpk(b, a)
    
    fig_pz, ax_pz = plt.subplots(figsize=(6, 4))
    
    # Unit circle
    unit_circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--', alpha=0.5)
    ax_pz.add_artist(unit_circle)
    
    # Poles and Zeros
    ax_pz.scatter(np.real(z), np.imag(z), s=50, marker='o', facecolors='none', edgecolors='b', label='Zeros')
    ax_pz.scatter(np.real(p), np.imag(p), s=50, marker='x', color='r', label='Poles')
    
    ax_pz.set_title("Pole-Zero Map")
    ax_pz.set_xlabel("Real")
    ax_pz.set_ylabel("Imaginary")
    ax_pz.grid(True, alpha=0.3)
    ax_pz.legend()
    ax_pz.set_aspect('equal')
    
    # Adjust limits to ensure unit circle is visible
    limit = max(1.5, np.max(np.abs(np.concatenate((z, p))))) if len(z) > 0 or len(p) > 0 else 1.5
    ax_pz.set_xlim(-limit, limit)
    ax_pz.set_ylim(-limit, limit)
    
    st.pyplot(fig_pz)

# Show coefficients
with st.expander("Filter Coefficients"):
    st.write("Numerator (b):", b)
    st.write("Denominator (a):", a)
