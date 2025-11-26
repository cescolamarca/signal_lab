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
st.markdown("### The DSP Workbench")

st.markdown("""
Welcome to **SignalLab**, an interactive environment for exploring Digital Signal Processing concepts.

### Modules

- **[Signal Composer](/Signal_Composer)**: Create and analyze custom signals.
- **[Standard Signals](/Standard_Signals)**: Explore fundamental waveforms.
- **[Convolution Demo](/Convolution_Demo)**: Visual step-by-step convolution.
- **[Frequency Domain](/Frequency_Domain)**: *Coming Soon* - FFT and Spectrograms.
- **[Filter Design](/Filter_Design)**: *Coming Soon* - Interactive Filter Design.

Select a module from the sidebar to begin.
""")

# Sidebar footer or info
st.sidebar.markdown("---")
st.sidebar.caption("SignalLab v2.0 | Designed for Signal Analysis")
