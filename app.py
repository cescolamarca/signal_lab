import streamlit as st
from signallab.style import set_custom_style

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="SignalLab DSP Workbench",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom premium style
set_custom_style(dark_mode=True)

st.title("SignalLab 📡")
st.markdown("### Interactive DSP Workbench for Signal Analysis")

st.markdown("""
Welcome to **SignalLab**, an interactive environment for exploring Digital Signal Processing concepts.
Built to support students in **Signal Analysis** courses with hands-on demonstrations.

---

### 🎯 Featured Demo
""")

# Featured demo card with highlight
st.info("""
**🎵 Audio Noise Removal** — See digital filtering in action!

- Generate noisy audio or upload your own WAV files
- Design FIR/IIR filters and apply them in real-time
- Compare before/after spectrograms with audio playback
- Measure actual noise reduction (SNR improvement)

👉 [Try Audio Processing](/Audio_Processing)
""")

st.markdown("""
---

### 📚 All Modules

| Module | Description |
|--------|-------------|
| **[Signal Composer](/Signal_Composer)** | Create signals using mathematical expressions |
| **[Standard Signals](/Standard_Signals)** | Explore fundamental waveforms with sampling demo |
| **[Convolution Demo](/Convolution_Demo)** | Visual step-by-step discrete convolution |
| **[Frequency Domain](/Frequency_Domain)** | FFT analysis with peak detection & spectrograms |
| **[Filter Design](/Filter_Design)** | Interactive IIR/FIR filter design with live demo |
| **[Audio Processing](/Audio_Processing)** | ⭐ Complete noise removal workflow |

Select a module from the sidebar to begin.
""")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.page_link("pages/06_Audio_Processing.py", label="🎵 Audio Processing", icon="⭐")
st.sidebar.page_link("pages/04_Frequency_Domain.py", label="📊 Frequency Domain")
st.sidebar.page_link("pages/05_Filter_Design.py", label="🎚️ Filter Design")

st.sidebar.markdown("---")
st.sidebar.caption("SignalLab v2.1 | Designed for Signal Analysis Education")
