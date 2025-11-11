"""
SignalLab - Interactive Signal Analysis Dashboard

This Streamlit application provides an interactive visualization of signal convolution.
It demonstrates the convolution operation step-by-step, showing:
- Input signals x(n) and h(n)
- Signal decomposition into unit impulses
- Shifted impulse responses
- Final convolution result y(n) = x(n) * h(n)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from signallab.plotting import (
    plot_signal,
    plot_signal_with_grid,
    plot_shifted_impulse_responses,
    plot_convolution_result
)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Configuration: Y-axis maximum value (adaptable by user)
st.sidebar.header("Plot Configuration")
YMAX = st.sidebar.number_input(
    "Y-axis maximum value",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Set the maximum value for the y-axis in all plots"
)

# ============================================================================
# LAYOUT: Two-column layout for x(n) and h(n)
# ============================================================================
col1, col2 = st.columns(2)

# ============================================================================
# LEFT COLUMN: Input Signal x(n)
# ============================================================================
with col1:
    st.subheader("Input Signal x(n)")

    # User input for signal x(n)
    x = np.fromstring(
        st.text_input("x(n) values (comma-separated, e.g., 1,2,1)", "1,2,1"),
        sep=","
    )
    start_x = int(st.number_input(
        label="Starting index for x(n)",
        value=0,
        step=1,
        help="The time index where x(n) begins"
    ))

    # Generate time indices for x(n)
    n_x = np.arange(start_x, start_x + len(x))

    # Plot the complete signal x(n)
    xlim_x = (start_x - 2, start_x + len(x) + 2)
    fig = plot_signal(n_x, x, "x(n)", xlim_x, YMAX, n_x, color='C0')
    st.pyplot(fig)

    # ========================================================================
    # Signal Decomposition: Show x(n) as sum of scaled unit impulses
    # ========================================================================
    st.markdown("### Signal Decomposition")
    st.markdown("Breaking down x(n) into scaled and shifted unit impulses:")

    for i in n_x:
        # Display mathematical notation for each impulse component
        if (i >= 0):
            st.latex(f"x({i})\\,\\delta(n - {i})")
        else:
            st.latex(f"x({i})\\,\\delta(n + {-1 * i})")

        # Plot the individual impulse component
        signal_value = x[i - start_x]  # Get the coefficient for this impulse
        fig = plot_signal_with_grid(
            [i], [signal_value],
            xlim_x, YMAX, n_x,
            color='C0'
        )
        st.pyplot(fig)

# ============================================================================
# RIGHT COLUMN: Impulse Response h(n) and Convolution Steps
# ============================================================================
with col2:
    st.subheader("Impulse Response h(n)")

    # User input for signal h(n)
    h = np.fromstring(
        st.text_input("h(n) values (comma-separated, e.g., 1,2,1)", "1,2,1"),
        sep=","
    )
    start_h = int(st.number_input(
        label="Starting index for h(n)",
        value=0,
        step=1,
        help="The time index where h(n) begins"
    ))

    # Generate base time indices for h(n)
    n_h0 = np.arange(start_h, start_h + len(h))

    # Plot the impulse response h(n)
    xlim_h = (start_h - 2, start_h + len(h) + 2)
    fig = plot_signal(n_h0, h, "h(n)", xlim_h, YMAX, n_h0, color='C1')
    st.pyplot(fig)

    # ========================================================================
    # Compute global axis range for convolution visualization
    # ========================================================================
    # The convolution y[n] will span from start_y to end_y
    start_y = start_x + start_h
    end_y = start_y + (len(x) + len(h) - 2)

    # Determine global axis limits to show all signals
    global_min = min(start_x, start_h, start_y)
    global_max = max(start_x + len(x) - 1, start_h + len(h) - 1, end_y)
    global_ticks = np.arange(global_min, global_max + 1)
    global_xlim = (global_min - 2, global_max + 2)

    # ========================================================================
    # Convolution Steps: Show individual responses x(i) * h(n - i)
    # ========================================================================
    plot_shifted_impulse_responses(n_x, x, start_x, h, n_h0, global_xlim, YMAX, global_ticks)

    # ========================================================================
    # Final Result: Convolution y(n) = x(n) * h(n)
    # ========================================================================
    plot_convolution_result(x, h, start_x, start_h, global_xlim, YMAX, global_ticks)
