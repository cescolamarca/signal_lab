import streamlit as st
import numpy as np
from signallab.composer import SignalComponent, generate_composite_signal
from signallab.plotting import plot_continuous_signal

st.set_page_config(page_title="Signal Composer", page_icon="🎼", layout="wide")

st.title("Signal Composer 🎼")
st.markdown("Compose a complex signal by summing multiple standard signals.")

# Initialize session state for components
if "components" not in st.session_state:
    st.session_state.components = []

# Sidebar for adding components
st.sidebar.header("Add Component")
with st.sidebar.form("add_component_form"):
    sig_type = st.selectbox(
        "Signal Type",
        ["Rectangular Pulse", "Triangular Pulse", "Sinc Function", "Heaviside Step", "Dirac Delta"]
    )
    amp = st.number_input("Amplitude", value=1.0, step=0.1)
    shift = st.number_input("Shift", value=0.0, step=0.5)
    
    scale = 1.0
    skew = 0.0
    
    if sig_type == "Rectangular Pulse":
        scale = st.number_input("Width", value=1.0, min_value=0.1, step=0.1)
    elif sig_type == "Triangular Pulse":
        scale = st.number_input("Width", value=1.0, min_value=0.1, step=0.1)
        skew = st.slider("Skew", -1.0, 1.0, 0.0, 0.1)
    elif sig_type == "Sinc Function":
        scale = st.number_input("Scale Factor", value=1.0, step=0.1)
    
    submitted = st.form_submit_button("Add Signal")
    if submitted:
        new_comp = SignalComponent(
            type_name=sig_type,
            amplitude=amp,
            shift=shift,
            scale=scale,
            skew=skew
        )
        st.session_state.components.append(new_comp)
        st.success(f"Added {sig_type}")

# Main Area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Composite Signal Plot")
    
    # Plot controls
    t_min = st.number_input("t_min", value=-5.0, step=1.0)
    t_max = st.number_input("t_max", value=5.0, step=1.0)
    resolution = st.slider("Resolution", 100, 2000, 1000)
    
    t = np.linspace(t_min, t_max, resolution)
    
    if st.session_state.components:
        y_total = generate_composite_signal(st.session_state.components, t)
        
        # Determine ylim
        ymin, ymax = np.min(y_total), np.max(y_total)
        margin = 0.1 * (ymax - ymin) if ymax != ymin else 0.5
        ylim = (ymin - margin, ymax + margin)
        
        fig = plot_continuous_signal(t, y_total, "Composite Signal", xlim=(t_min, t_max), ylim=ylim)
        st.pyplot(fig)
        
        # Equation
        st.subheader("Mathematical Expression")
        latex_parts = [c.to_latex() for c in st.session_state.components]
        latex_eq = " + ".join(latex_parts).replace("+ -", "- ")
        st.latex(f"y(t) = {latex_eq}")
    else:
        st.info("Add signals from the sidebar to see the plot.")

with col2:
    st.subheader("Components")
    if st.session_state.components:
        for i, comp in enumerate(st.session_state.components):
            with st.expander(f"{i+1}. {comp.type_name}", expanded=True):
                st.write(f"**Amp:** {comp.amplitude}")
                st.write(f"**Shift:** {comp.shift}")
                if comp.type_name in ["Rectangular Pulse", "Triangular Pulse"]:
                    st.write(f"**Width:** {comp.scale}")
                if comp.type_name == "Triangular Pulse":
                    st.write(f"**Skew:** {comp.skew}")
                if comp.type_name == "Sinc Function":
                    st.write(f"**Scale:** {comp.scale}")
                
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.components.pop(i)
                    st.rerun()
    else:
        st.write("No components added.")

    if st.session_state.components:
        if st.button("Clear All"):
            st.session_state.components = []
            st.rerun()
