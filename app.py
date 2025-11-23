import streamlit as st

st.set_page_config(
    page_title="SignalLab",
    page_icon="📡",
    layout="wide"
)

st.title("SignalLab 📡")
st.markdown("""
### Welcome to SignalLab

This application is a playground for signal processing concepts.

#### Available Modules:

- **[Signal Composer](/Signal_Composer)**: Create complex signals by summing standard waveforms.
- **[Standard Signals](/Standard_Signals)**: Visualize individual standard signals like Rect, Tri, Sinc, etc.
- **[Convolution Demo](/Convolution_Demo)**: Interactive demonstration of discrete-time convolution.

Select a module from the sidebar to get started.
""")
