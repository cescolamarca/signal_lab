import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from signallab.style import set_custom_style, get_plot_colors
from signallab.utils import parse_expression, format_number

st.markdown("# Signal Composer")
st.markdown("Generate signals using mathematical expressions and analyze them.")

# Sidebar controls
st.sidebar.header("Signal Configuration")
duration = st.sidebar.slider("Duration (s)", 0.1, 10.0, 1.0)
fs = st.sidebar.slider("Sampling Rate (Hz)", 10, 1000, 100)

t = np.linspace(0, duration, int(duration * fs))

# Expression Input
st.subheader("Signal Definition")
default_expr = "sin(2*pi*5*t) + 0.5*sin(2*pi*12*t)"
expression = st.text_input("Enter Expression (use 't' as time variable)", value=default_expr)

try:
    y = parse_expression(expression, t)
    
    # Plotting
    colors = get_plot_colors()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, color=colors[0], label="Signal y(t)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Signal: {expression}")
    ax.legend()
    
    st.pyplot(fig)
    
    # Stats
    st.markdown("### Signal Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", format_number(np.mean(y)))
    col2.metric("Max", format_number(np.max(y)))
    col3.metric("Min", format_number(np.min(y)))
    
except Exception as e:
    st.error(f"Error parsing expression: {e}")
    st.info("Supported functions: sin, cos, tan, exp, sqrt, abs. Constant: pi.")
