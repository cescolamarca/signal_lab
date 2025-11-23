import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from signallab.signals import rectpuls, tripuls, sinc, heaviside, dirac
from signallab.plotting import plot_continuous_signal

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SignalLab - Standard Signals",
    page_icon="📈",
    layout="wide"
)

st.title("Standard Signal Generation")
st.markdown("""
This module allows you to generate and visualize standard signals:
- **Rectangular Pulse**: `rectpuls(t)`
- **Triangular Pulse**: `tripuls(t)`
- **Sinc Function**: `sinc(t)`
- **Heaviside Step**: `heaviside(t)`
- **Dirac Delta**: `dirac(t)`

You can adjust parameters to shift and scale these signals.
""")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.header("Signal Parameters")

signal_type = st.sidebar.selectbox(
    "Select Signal Type",
    ["Rectangular Pulse", "Triangular Pulse", "Sinc Function", "Heaviside Step", "Dirac Delta"]
)

# Common parameters
t_min = st.sidebar.number_input("Time Start (t_min)", value=-5.0, step=0.5)
t_max = st.sidebar.number_input("Time End (t_max)", value=5.0, step=0.5)
num_points = st.sidebar.number_input("Number of Points", value=1000, step=100)

t = np.linspace(t_min, t_max, int(num_points))

# Signal specific parameters
st.sidebar.markdown("---")
st.sidebar.subheader(f"{signal_type} Parameters")

y = None
title = ""
code_str = ""

if signal_type == "Rectangular Pulse":
    width = st.sidebar.number_input("Width (w)", value=1.0, step=0.1)
    shift = st.sidebar.number_input("Shift (center)", value=0.0, step=0.1)
    
    # Apply shift: t -> t - shift
    y = rectpuls(t - shift, w=width)
    title = f"Rectangular Pulse (w={width}, shift={shift})"
    code_str = f"y = rectpuls(t - {shift}, w={width})"

elif signal_type == "Triangular Pulse":
    width = st.sidebar.number_input("Width (w)", value=1.0, step=0.1)
    skew = st.sidebar.slider("Skew (s)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    shift = st.sidebar.number_input("Shift (center)", value=0.0, step=0.1)
    
    y = tripuls(t - shift, w=width, s=skew)
    title = f"Triangular Pulse (w={width}, s={skew}, shift={shift})"
    code_str = f"y = tripuls(t - {shift}, w={width}, s={skew})"

elif signal_type == "Sinc Function":
    scale = st.sidebar.number_input("Scale Factor (a in sinc(a*t))", value=1.0, step=0.1)
    shift = st.sidebar.number_input("Shift (center)", value=0.0, step=0.1)
    
    # MATLAB sinc is sin(pi*t)/(pi*t). Scaling t scales the frequency.
    # sinc(a*(t-shift))
    y = sinc(scale * (t - shift))
    title = f"Sinc Function (scale={scale}, shift={shift})"
    code_str = f"y = sinc({scale} * (t - {shift}))"

elif signal_type == "Heaviside Step":
    shift = st.sidebar.number_input("Shift (center)", value=0.0, step=0.1)
    
    y = heaviside(t - shift)
    title = f"Heaviside Step (shift={shift})"
    code_str = f"y = heaviside(t - {shift})"

elif signal_type == "Dirac Delta":
    shift = st.sidebar.number_input("Shift (center)", value=0.0, step=0.1)
    
    y = dirac(t - shift)
    title = f"Dirac Delta (shift={shift})"
    code_str = f"y = dirac(t - {shift})"

# ============================================================================
# MAIN DISPLAY
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Visualization")
    if y is not None:
        # Determine appropriate y-limits
        ymin, ymax = np.min(y), np.max(y)
        margin = 0.1 * (ymax - ymin) if ymax != ymin else 0.5
        ylim = (ymin - margin, ymax + margin)
        
        fig = plot_continuous_signal(t, y, title, xlim=(t_min, t_max), ylim=ylim)
        st.pyplot(fig)

with col2:
    st.subheader("Details")
    st.code(code_str, language="python")
    st.markdown("### Values")
    st.write(f"**Max Value:** {np.max(y):.4f}")
    st.write(f"**Min Value:** {np.min(y):.4f}")
    st.write(f"**Energy:** {np.sum(y**2) * (t[1]-t[0]):.4f}")

# ============================================================================
# INSTRUCTIONS
# ============================================================================
st.markdown("---")
st.markdown("""
### How to use
1. Select a signal type from the sidebar.
2. Adjust the time range and resolution.
3. Modify signal-specific parameters like width, skew, or shift.
4. Observe the plot and the corresponding Python code.
""")
