import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from signallab.style import set_custom_style, get_plot_colors
from signallab.signals import get_signal_function
from signallab.utils import format_number

st.markdown("# Standard Signals")
st.markdown("Explore standard signal waveforms and their properties.")

# Signal Selection
signal_type = st.selectbox(
    "Select Signal Type",
    ["Rectangular Pulse", "Triangular Pulse", "Sinc Function", "Heaviside Step", "Dirac Delta"]
)

# Parameters
col1, col2 = st.columns(2)
with col1:
    duration = st.slider("Time Span (s)", 1.0, 10.0, 4.0)
    fs = st.slider("Sampling Rate (Hz)", 10, 200, 100)

t = np.linspace(-duration/2, duration/2, int(duration * fs))

# Signal Generation
sig_func = get_signal_function(signal_type)
params = {}

with col2:
    if signal_type in ["Rectangular Pulse", "Triangular Pulse"]:
        width = st.slider("Width", 0.1, duration, 1.0)
        params['w'] = width
    if signal_type == "Triangular Pulse":
        skew = st.slider("Skew", -1.0, 1.0, 0.0)
        params['s'] = skew

if sig_func:
    y = sig_func(t, **params)
    
    # Plotting
    colors = get_plot_colors()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Continuous-like representation
    ax.plot(t, y, color=colors[1], label=signal_type, linewidth=2)
    
    # Stem plot for discrete view (optional or overlay)
    show_samples = st.checkbox("Show Samples (Discrete View)", value=False)
    if show_samples:
        marker, stemlines, baseline = ax.stem(t, y, linefmt=colors[1], markerfmt='o', basefmt=" ")
        plt.setp(stemlines, 'color', colors[1], 'linewidth', 1, 'alpha', 0.5)
        plt.setp(marker, 'color', colors[1], 'markersize', 4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{signal_type} Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Sampling / Aliasing Demo Integration
    st.markdown("### Sampling & Aliasing Demo")
    st.info("Reduce the sampling rate to see aliasing effects (jagged lines or incorrect reconstruction).")
    
    # Resample
    demo_fs = st.slider("Demo Sampling Rate (Hz)", 1, 50, 10, key="demo_fs")
    t_sampled = np.linspace(-duration/2, duration/2, int(duration * demo_fs))
    y_sampled = sig_func(t_sampled, **params)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    # Ghost of original
    ax2.plot(t, y, color='gray', alpha=0.3, label="Original (High Fs)", linestyle='--')
    # Sampled
    marker, stemlines, baseline = ax2.stem(t_sampled, y_sampled, linefmt=colors[2], markerfmt='o', basefmt=" ", label=f"Sampled @ {demo_fs}Hz")
    plt.setp(marker, color=colors[2])
    plt.setp(stemlines, color=colors[2])
    
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Sampling Visualization")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.error("Signal function not found.")
