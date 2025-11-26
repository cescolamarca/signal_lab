import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from signallab.style import get_plot_colors

st.markdown("# Convolution Demo")
st.markdown("Visualize discrete-time convolution $y[n] = x[n] * h[n]$ step-by-step.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Signal x[n]")
    x_str = st.text_input("Values (comma separated)", "1, 2, 3")
    x_origin = st.number_input("Origin Index (n=0)", value=0, key="x_origin")
    
with col2:
    st.subheader("Impulse Response h[n]")
    h_str = st.text_input("Values (comma separated)", "1, 1, 1")
    h_origin = st.number_input("Origin Index (n=0)", value=0, key="h_origin")

try:
    x = np.array([float(v) for v in x_str.split(',')])
    h = np.array([float(v) for v in h_str.split(',')])
    
    # Indices
    start_x = -int(x_origin)
    start_h = -int(h_origin)
    n_x = np.arange(start_x, start_x + len(x))
    n_h = np.arange(start_h, start_h + len(h))
    
    # Calculate global plotting parameters
    start_y = start_x + start_h
    end_y = start_y + len(x) + len(h) - 1
    global_xlim = (min(start_x, start_h, start_y) - 1, max(start_x + len(x), start_h + len(h), end_y) + 1)
    ymax = max(np.max(np.abs(x)), np.max(np.abs(h)), np.max(np.abs(np.convolve(x, h)))) * 1.2
    global_ticks = np.arange(global_xlim[0], global_xlim[1] + 1)
    
    colors = get_plot_colors()
    
    # Plot input signals
    st.markdown("### Input Signals")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_x, ax_x = plt.subplots(figsize=(5, 3))
        marker, stemlines, baseline = ax_x.stem(n_x, x, basefmt=" ")
        plt.setp(marker, color=colors[0])
        plt.setp(stemlines, color=colors[0])
        ax_x.set_xlim(global_xlim)
        ax_x.set_ylim(-ymax, ymax)
        ax_x.set_xticks(global_ticks)
        ax_x.set_xlabel("n")
        ax_x.set_title("x[n]")
        ax_x.grid(True, alpha=0.3)
        st.pyplot(fig_x)
    
    with col2:
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        marker, stemlines, baseline = ax_h.stem(n_h, h, basefmt=" ")
        plt.setp(marker, color=colors[1])
        plt.setp(stemlines, color=colors[1])
        ax_h.set_xlim(global_xlim)
        ax_h.set_ylim(-ymax, ymax)
        ax_h.set_xticks(global_ticks)
        ax_h.set_xlabel("n")
        ax_h.set_title("h[n]")
        ax_h.grid(True, alpha=0.3)
        st.pyplot(fig_h)
    
    # Convolution steps with two-column layout
    st.markdown("### Convolution Steps")
    st.markdown("Each component of x[n] produces a shifted and scaled version of h[n]:")
    
    for i, (n_val, x_val) in enumerate(zip(n_x, x)):
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Show the x[n] sample being picked
            st.latex(f"x[{n_val}] = {x_val:.2f}")
            fig_x_sample, ax_x_sample = plt.subplots(figsize=(5, 3))
            # Highlight the current sample
            marker, stemlines, baseline = ax_x_sample.stem(n_x, x, basefmt=" ")
            plt.setp(marker, color=colors[0], alpha=0.3)
            plt.setp(stemlines, color=colors[0], alpha=0.3)
            # Highlight current sample
            marker_hl, stemlines_hl, baseline_hl = ax_x_sample.stem([n_val], [x_val], basefmt=" ", linefmt=colors[0], markerfmt='o')
            plt.setp(marker_hl, markersize=8, color=colors[0])
            plt.setp(stemlines_hl, color=colors[0])
            ax_x_sample.set_xlim(global_xlim)
            ax_x_sample.set_ylim(-ymax, ymax)
            ax_x_sample.set_xticks(global_ticks)
            ax_x_sample.set_xlabel("n")
            ax_x_sample.grid(True, alpha=0.3)
            st.pyplot(fig_x_sample)
        
        with col_right:
            # Show the shifted and scaled h[n]
            if n_val >= 0:
                st.latex(f"x[{n_val}] \\cdot h[n - {n_val}]")
            else:
                st.latex(f"x[{n_val}] \\cdot h[n + {-n_val}]")
            
            # Compute shifted indices: h(n - i) means shift h by +i
            n_hi = n_h + n_val
            h_shifted = x_val * h
            
            fig_h_shifted, ax_h_shifted = plt.subplots(figsize=(5, 3))
            marker, stemlines, baseline = ax_h_shifted.stem(n_hi, h_shifted, basefmt=" ")
            plt.setp(marker, color=colors[1])
            plt.setp(stemlines, color=colors[1])
            ax_h_shifted.set_xlim(global_xlim)
            ax_h_shifted.set_ylim(-ymax, ymax)
            ax_h_shifted.set_xticks(global_ticks)
            ax_h_shifted.set_xlabel("n")
            ax_h_shifted.grid(True, alpha=0.3)
            st.pyplot(fig_h_shifted)
    
    # Convolution result
    st.markdown("### Convolution Result")
    st.markdown("The output y[n] is the sum of all shifted impulse responses:")
    
    y = np.convolve(x, h)
    n_y = np.arange(start_y, start_y + len(y))
    
    fig_result, ax_result = plt.subplots(figsize=(10, 4))
    marker, stemlines, baseline = ax_result.stem(n_y, y, basefmt=" ")
    plt.setp(marker, color=colors[2])
    plt.setp(stemlines, color=colors[2])
    ax_result.set_xlim(global_xlim)
    ax_result.set_ylim(-ymax, ymax)
    ax_result.set_xticks(global_ticks)
    ax_result.set_xlabel("n")
    ax_result.set_title("y[n] = x[n] * h[n]")
    ax_result.grid(True, alpha=0.3)
    st.pyplot(fig_result)

except ValueError:
    st.error("Invalid input. Please enter comma-separated numbers.")

