# SignalLab - Interactive DSP Workbench
#### Video Demo: <URL HERE>
#### Description:

---

## What is SignalLab?

SignalLab is an interactive web-based educational platform for exploring Digital Signal Processing (DSP) concepts, built specifically for university-level "Signal Analysis" courses. The project bridges the gap between theoretical DSP knowledge and practical, hands-on understanding through visual, interactive demonstrations. As a student who found traditional DSP education too abstract, I created SignalLab to provide the kind of tool I wished I had when first learning these concepts.

The application offers three core modules: a Signal Composer for creating custom signals using mathematical expressions, a Standard Signals library with sampling demonstrations, and an innovative Convolution Demo that visualizes the convolution operation step-by-step. Each module is designed with educational clarity in mind, using premium dark-themed visualizations that reduce eye strain during extended study sessions.

## Why This Project?

Digital Signal Processing is a foundational subject in electrical engineering, computer science, and related fields. However, students often struggle to develop intuition for abstract concepts like convolution, sampling, and aliasing when taught purely through equations and static textbook diagrams. While MATLAB and similar tools exist, they often have steep learning curves and lack the interactivity needed for exploratory learning.

SignalLab addresses this gap by providing an intuitive, web-based interface where students can immediately see the results of changing signal parameters, experiment with different sampling rates to observe aliasing, and watch convolution unfold sample-by-sample. The project aims to be a portfolio-worthy demonstration of full-stack Python development, UI/UX design for education, and domain expertise in signal processing.

## Technical Architecture

SignalLab is built as a multi-page Streamlit application, leveraging Python's scientific computing ecosystem. The architecture follows a clear separation of concerns: a core library (`signallab/`) provides signal generation, composition, and plotting utilities, while individual page modules (`pages/`) implement specific educational experiences. This modular design makes the codebase maintainable and extensible for future features like FFT analysis or filter design.

## File-by-File Documentation

### `app.py` - Application Entry Point

This is the main entry point for the Streamlit application. It configures the page settings (title, icon, wide layout), applies the custom dark theme via `set_custom_style()`, and displays the home page with navigation links to each module. The file is intentionally minimal (35 lines) as Streamlit automatically handles routing to page modules in the `pages/` directory. Key design choice: using Streamlit's built-in multi-page app structure rather than manual routing keeps the code clean and leverages framework conventions.

### `requirements.txt` - Python Dependencies

Lists the three core dependencies: `streamlit` (web framework), `numpy` (numerical computing), and `matplotlib` (visualization). Keeping dependencies minimal was a deliberate choice to ensure the project remains lightweight, easy to install, and free from complex dependency conflicts. Each library serves a critical purpose that cannot easily be replaced.

### Pages Directory - Streamlit Multi-Page Modules

Streamlit's multi-page app feature automatically discovers Python files in the `pages/` directory and creates navigation. Files are prefixed with numbers (`01_`, `02_`, `03_`) to control their order in the sidebar.

#### `01_Signal_Composer.py` - Expression-Based Signal Generation

This module allows users to generate custom signals by typing mathematical expressions like `sin(2*pi*5*t) + 0.5*cos(2*pi*10*t)`. It uses the `parse_and_evaluate_expression()` function from `signallab.utils` to safely evaluate user input. The expression parser supports standard functions (`sin`, `cos`, `exp`, `sqrt`, `abs`) while blocking potentially dangerous code execution.

**Design rationale**: Initially, I considered using separate sliders for each signal parameter, but this approach limited users to predefined signal types. The expression-based approach gives advanced users complete freedom while still being accessible to beginners through examples. The parser implementation uses Python's `ast` module for safe evaluation rather than `eval()`, which would be a security risk.

#### `02_Standard_Signals.py` - Fundamental Waveforms Library

Provides interactive visualizations of five fundamental signal types: Rectangular Pulse, Triangular Pulse, Sinc Function, Heaviside Step, and Dirac Delta. Each signal has customizable parameters (amplitude, shift, width, skew) controlled through Streamlit sliders.

The module includes a **Sampling & Aliasing Demo** where users can adjust the sampling rate and visually observe aliasing effects. This demonstrates the Nyquist theorem in action - when the sampling rate is too low, high-frequency components appear as false low-frequency components. The visualization shows both the continuous signal and discrete samples overlaid, making the phenomenon immediately apparent.

**Implementation note**: The Dirac Delta function is approximated as a very narrow Gaussian pulse, as true mathematical delta functions cannot be plotted. This is a common practical approach in discrete-time signal processing education.

#### `03_Convolution_Demo.py` - Interactive Convolution Visualization

This is the most pedagogically innovative module. Convolution is typically one of the most difficult DSP concepts for students to grasp, as the mathematical definition (a sum of products with time-reversed, shifted signals) is not intuitive. 

The module presents convolution as a step-by-step process using a **two-column layout**:
- **Left column**: Shows which sample of the input signal `x[n]` is currently being processed
- **Right column**: Displays the corresponding impulse response `h[n]` shifted and scaled by that sample

This visualization directly implements the interpretation of convolution as a weighted sum of shifted impulse responses, which is more intuitive than the standard flip-and-slide explanation. Users can step through each sample and see the partial output build up over time.

**Design choice**: The two-column approach was chosen after considering alternatives like animation (too fast for learning) or static diagrams (not interactive enough). The step-by-step format lets students control the pace and focus on understanding each contribution to the final output.

### SignalLab Library - Core Modules

The `signallab/` directory contains the core library that powers all page modules. This separation allows for code reuse and cleaner page implementations.

#### `signals.py` - Signal Generation Functions

Implements the mathematical definitions of all standard signals as NumPy functions:
- `rectpuls(t, w)`: Rectangular pulse of width `w`
- `tripuls(t, w, s)`: Triangular pulse with width `w` and skew `s` (-1 to 1)
- `sinc_function(t)`: Normalized sinc function, $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$
- `heaviside(t)`: Unit step function (0 for t < 0, 1 for t ≥ 0)
- `dirac_delta(t)`: Approximated as narrow Gaussian

Also includes `get_signal_function()`, a factory function that returns the appropriate signal function by name, enabling dynamic signal selection in the UI.

**Implementation detail**: All functions use NumPy's vectorized operations for efficiency, allowing plots to be generated from thousands of time samples without performance issues.

#### `composer.py` - Signal Composition Framework

Defines the `SignalComponent` dataclass, which represents a single signal with parameters (type, amplitude, shift, scale, skew). Each component can evaluate itself over a time array and generate LaTeX representation for display.

The `generate_composite_signal()` function sums multiple components, enabling users to build complex signals from primitives. This design uses object-oriented principles to encapsulate signal behavior while keeping the code simple and readable.

**Design rationale**: The dataclass approach (versus plain dictionaries) provides type safety and auto-generated `__init__` methods. The LaTeX generation allows displaying mathematical notation in the UI, which is important for educational contexts.

#### `utils.py` - Expression Parsing and Formatting

Contains two key utilities:
1. **`parse_and_evaluate_expression()`**: Safely evaluates mathematical expressions using Python's `ast` (Abstract Syntax Tree) module. Only allows mathematical operations and whitelisted functions, preventing code injection attacks.
2. **`format_number_precise()`**: Formats floating-point numbers with intelligent precision, avoiding scientific notation and unnecessary trailing zeros. This ensures plot labels remain readable.

**Security consideration**: Using `ast.parse()` and `ast.NodeVisitor` to validate expressions before evaluation is significantly safer than `eval()`, which would allow arbitrary code execution. The parser explicitly whitelists allowed node types and function names.

#### `style.py` - Custom Matplotlib Theming

Configures Matplotlib with a premium dark theme that matches Streamlit's dark mode aesthetic. Sets custom colors (teal accent `#00ADB5`), background colors (`#0E1117`), grid styles, and font preferences.

**Design choice**: The dark theme was chosen for reduced eye strain and modern aesthetics. The original plan used the 'Inter' font, but this was changed to 'Segoe UI' (Windows system font) for better compatibility across systems without requiring font installation.

#### `plotting.py` - Reusable Plotting Utilities

Provides helper functions for common plotting patterns: `plot_signal_with_samples()` for time-domain plots with optional discrete samples, and `create_stem_plot()` for discrete-time visualizations.

Centralizing plotting logic here ensures consistent visual style across all modules and reduces code duplication. The functions handle common tasks like grid setup, axis labeling, and color management.

## Key Design Decisions

### Why Streamlit?

I chose Streamlit over alternatives like Flask/Django or Dash for several reasons:
1. **Rapid prototyping**: Streamlit allows building interactive apps with minimal boilerplate
2. **Built-in widgets**: Sliders, buttons, and layouts are trivial to implement
3. **Automatic reactivity**: UI updates automatically when inputs change, no callback wiring needed
4. **Easy deployment**: Can be deployed to Streamlit Cloud for free, making the project accessible to students worldwide

The tradeoff is less control over UI customization compared to HTML/CSS/JavaScript, but for an educational tool, the development speed advantage outweighs this limitation.

### Modular Architecture

Separating core functionality (`signallab/`) from UI pages (`pages/`) and the main entry point (`app.py`) follows software engineering best practices. This makes the code:
- **Testable**: Core functions can be unit tested independently
- **Reusable**: Signal functions could be used in other projects
- **Maintainable**: Changes to UI don't require touching signal generation logic
- **Extensible**: New modules can be added by creating new page files

### Expression Parser vs. GUI Builder

For the Signal Composer, I debated between a drag-and-drop GUI builder (like Simulink) versus text-based expressions. I chose expressions because:
- **Efficiency**: Power users can type expressions faster than building graphical flowcharts
- **Precision**: Exact parameter values are easier to specify via text
- **Familiarity**: Students already know mathematical notation
- **Simplicity**: Implementing a visual builder would require complex state management

The tradeoff is a steeper initial learning curve for non-technical users, but the module includes examples to guide beginners.

## Technologies Used

- **[Streamlit](https://streamlit.io/)**: Python web framework for data science applications
- **[NumPy](https://numpy.org/)**: Fundamental package for scientific computing with Python
- **[Matplotlib](https://matplotlib.org/)**: Comprehensive library for creating static, animated, and interactive visualizations

## Installation and Usage

### Prerequisites
- Python 3.10 or higher (tested on Python 3.13)
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cescolamarca/signal_lab.git
   cd signal_lab
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   py -3.13 -m venv .venv
   
   # macOS/Linux
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # Windows
   .\.venv\Scripts\activate
   
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

## Educational Value

SignalLab demonstrates concepts critical to Digital Signal Processing education:

- **Time-domain signal representation**: Understanding signals as functions of time
- **Convolution theory**: The fundamental operation in linear time-invariant systems
- **Sampling and aliasing**: Core requirements for converting continuous signals to digital form
- **Signal composition**: Building complex signals from elementary components

The interactive approach supports active learning - students learn by doing rather than passively reading.

## Future Enhancements

While the current version focuses on time-domain analysis and convolution, future modules could include:

- **Frequency Domain Analysis**: FFT computation and spectrograms
- **Filter Design**: Interactive IIR and FIR filter design with frequency response plots
- **Z-Transform Visualization**: Pole-zero plots and system stability analysis
- **Real-time Audio Processing**: Apply DSP concepts to live microphone input

## Project Information

- **Author**: Francesco La Marca
- **GitHub**: [cescolamarca](https://github.com/cescolamarca)
- **edX**: cescolamarca
- **Location**: Naples (NA), Italy
- **Date**: November 27, 2024

---

**This project was created as the final project for CS50x 2024.**
