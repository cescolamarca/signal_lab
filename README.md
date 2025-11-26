# SignalLab - DSP Workbench

An interactive web application for exploring Digital Signal Processing (DSP) concepts, built with Python and Streamlit. Designed as an educational tool for university-level "Signal Analysis" courses.

## ✨ Features

### 📊 Signal Composer
- Generate custom signals using mathematical expressions (e.g., `sin(2*pi*5*t) + 0.5*cos(2*pi*10*t)`)
- Support for standard mathematical functions: `sin`, `cos`, `exp`, `sqrt`, `abs`
- Real-time signal visualization

### 🎯 Standard Signals
- Visualize fundamental waveforms: Rectangular Pulse, Triangular Pulse, Sinc, Heaviside Step, Dirac Delta
- Interactive parameters for signal customization (width, skew, etc.)
- **Sampling & Aliasing Demo**: Adjust sampling rates to visualize aliasing effects on any standard signal

### 🔄 Convolution Demo
- Step-by-step visualization of discrete-time convolution
- **Two-column layout**: See which x[n] sample multiplies each shifted h[n] response
- Understand convolution as a linear combination of shifted impulse responses


## 🚀 Getting Started

### Prerequisites

- Python 3.13+ (recommended) or Python 3.10+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cescolamarca/signallab.git
   cd signallab
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   py -3.13 -m venv .venv_313
   
   # macOS/Linux
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # Windows
   .\.venv_313\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
signallab/
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── pages/                    # Streamlit multi-page modules
│   ├── 01_Signal_Composer.py
│   ├── 02_Standard_Signals.py
│   └── 03_Convolution_Demo.py
└── signallab/                # Core library
    ├── signals.py            # Signal generation functions
    ├── utils.py              # Utilities (expression parsing, formatting)
    ├── style.py              # Custom Matplotlib theming
    └── plotting.py           # Plotting helper functions
```

## 🛠️ Technologies

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Matplotlib](https://matplotlib.org/)**: Data visualization

## 📚 Educational Value

This project demonstrates:
- Time-domain signal analysis
- Convolution theory and visualization
- Sampling theorem and aliasing
- Interactive pedagogical tools for DSP education

## 🎓 Use Cases

- **University Courses**: Supplement lectures in Signal Processing, Communications, or Control Systems
- **Self-Study**: Interactive exploration of DSP concepts
- **Portfolio Project**: Demonstrates full-stack Python development and UI/UX design

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the MIT License.
