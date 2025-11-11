"""
Plotting functions for SignalLab.

This module contains functions for visualizing discrete-time signals
and the convolution process.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def plot_signal(n_values, signal_values, title, xlim, ylim, xticks, color='C0'):
    """
    Plot a discrete-time signal using stem plot.
    
    Args:
        n_values: Array of time indices
        signal_values: Array of signal values
        title: Title for the plot
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
        xticks: Array of x-axis tick positions
        color: Color specification for the plot (default: 'C0')
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots()
    linefmt = f"{color}-" if color != 'C0' else 'C0-'
    markerfmt = f"{color}o" if color != 'C0' else 'C0o'
    ax.stem(n_values, signal_values, basefmt=" ", linefmt=linefmt, markerfmt=markerfmt)
    ax.set_xlim(xlim)
    ax.set_ylim(ymax=ylim)
    ax.set_xticks(xticks)
    ax.set_xlabel("n")
    ax.set_title(title)
    return fig


def plot_signal_with_grid(n_values, signal_values, xlim, ylim, xticks, color='C0'):
    """
    Plot a discrete-time signal with grid enabled.
    
    Args:
        n_values: Array of time indices
        signal_values: Array of signal values
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
        xticks: Array of x-axis tick positions
        color: Color specification for the plot (default: 'C0')
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots()
    linefmt = f"{color}-" if color != 'C0' else 'C0-'
    markerfmt = f"{color}o" if color != 'C0' else 'C0o'
    ax.stem(n_values, signal_values, basefmt=" ", linefmt=linefmt, markerfmt=markerfmt)
    ax.set_xlim(xlim)
    ax.set_ylim(ymax=ylim)
    ax.grid(True)
    ax.set_xticks(xticks)
    return fig


def plot_shifted_impulse_responses(n_x, x, start_x, h, n_h0, global_xlim, ymax, global_ticks):
    """
    Plot the shifted and scaled impulse responses for each component of x(n).
    
    This function visualizes the convolution process by showing how each sample
    of x(n) produces a shifted and scaled version of h(n).
    
    Args:
        n_x: Array of time indices for signal x(n)
        x: Array of signal x(n) values
        start_x: Starting index of signal x(n)
        h: Array of impulse response h(n) values
        n_h0: Base time indices for h(n)
        global_xlim: Tuple of (xmin, xmax) for global x-axis limits
        ymax: Maximum value for y-axis
        global_ticks: Array of global x-axis tick positions
    """
    st.markdown("### Convolution Steps")
    st.markdown("Each component of x(n) produces a shifted and scaled version of h(n):")

    for i in n_x:
        # Get the scaling coefficient from x(n)
        coeff = x[i - start_x]

        # Display mathematical notation for this shifted impulse response
        if (i >= 0):
            st.latex(f"x({i})\\,h(n - {i})")
        else:
            st.latex(f"x({i})\\,h(n + {-1 * i})")

        # Compute shifted indices: h(n - i) means shift h by +i
        n_hi = n_h0 + i

        # Plot the scaled and shifted impulse response
        fig = plot_signal_with_grid(
            n_hi, coeff * h,
            global_xlim, ymax, global_ticks,
            color='C1'
        )
        st.pyplot(fig)


def plot_convolution_result(x, h, start_x, start_h, global_xlim, ymax, global_ticks):
    """
    Compute and plot the convolution result y(n) = x(n) * h(n).
    
    Args:
        x: Array of signal x(n) values
        h: Array of impulse response h(n) values
        start_x: Starting index of signal x(n)
        start_h: Starting index of signal h(n)
        global_xlim: Tuple of (xmin, xmax) for global x-axis limits
        ymax: Maximum value for y-axis
        global_ticks: Array of global x-axis tick positions
    """
    st.markdown("### Convolution Result")
    st.markdown("The output y(n) is the sum of all shifted impulse responses:")

    # Compute convolution using NumPy
    y = np.convolve(x, h)

    # Determine time indices for convolution result
    start_y = start_x + start_h
    n_y = np.arange(start_y, start_y + len(y))

    # Plot the final convolution result
    fig = plot_signal(n_y, y, "y(n) = x(n) * h(n)", global_xlim, ymax, global_ticks, color='C3')
    st.pyplot(fig)
