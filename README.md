# SignalLab — Interactive DSP Workbench

SignalLab is a web-based DSP workbench built with Streamlit to help students build intuition for core signal analysis topics through interactive, visual demos.

## Features

### 1) Signal Composer
Create custom signals by typing mathematical expressions, e.g.
- `sin(2*pi*5*t) + 0.5*cos(2*pi*10*t)`

Expressions are evaluated safely (no `eval`) using an AST-based whitelist.

### 2) Standard Signals + Sampling/Aliasing
Interact with common waveforms and parameters:
- Rectangular pulse, triangular pulse, sinc, Heaviside step, Dirac delta (approximated)
- Sampling controls to visualize aliasing and the Nyquist limit

### 3) Convolution Demo (Step-by-step)
A guided visualization of discrete-time convolution as a **weighted sum of shifted impulse responses**:
- Shows the current input sample `x[n]`
- Shows the corresponding shifted/scaled `h[n]`
- Builds the output incrementally

## Libraries used
- Streamlit
- NumPy
- Matplotlib

## Project Structure

- `app.py` — Streamlit entry point (page config, theme/style, home)
- `pages/` — Streamlit multi-page modules  
  - `01_Signal_Composer.py`
  - `02_Standard_Signals.py`
  - `03_Convolution_Demo.py`
- `signallab/` — reusable core library
  - `signals.py` — standard signal definitions + signal factory
  - `composer.py` — signal components + composite generation
  - `plotting.py` — shared plotting helpers
  - `utils.py` — safe expression parsing + numeric formatting
  - `style.py` — Matplotlib dark theme styling
- `requirements.txt` — dependencies

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.13)

### Setup
```bash
git clone https://github.com/cescolamarca/signal_lab.git
cd signal_lab

python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
