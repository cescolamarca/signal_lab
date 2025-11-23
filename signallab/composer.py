from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from signallab.signals import get_signal_function

@dataclass
class SignalComponent:
    """
    Represents a single component of a composite signal.
    """
    type_name: str
    amplitude: float = 1.0
    shift: float = 0.0
    scale: float = 1.0 # For width or frequency scaling
    skew: float = 0.0 # Specific to triangular pulse

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate this signal component over time t.
        """
        func = get_signal_function(self.type_name)
        if func is None:
            return np.zeros_like(t)
        
        # Apply shift: t' = t - shift
        t_shifted = t - self.shift
        
        # Apply function specific logic
        # Note: Scale interpretation depends on function
        # rectpuls(t, w) -> w is width.
        # tripuls(t, w, s) -> w is width.
        # sinc(a*t) -> a is scale factor.
        # heaviside(t) -> scale doesn't usually apply to time, but maybe width? No.
        
        y = np.zeros_like(t)
        
        if self.type_name == "Rectangular Pulse":
            y = func(t_shifted, w=self.scale)
        elif self.type_name == "Triangular Pulse":
            y = func(t_shifted, w=self.scale, s=self.skew)
        elif self.type_name == "Sinc Function":
            # For sinc, let's assume scale is the multiplier 'a' in sinc(a*t)
            # If user wants width W of main lobe, a = 2/W? 
            # Standard definition: sinc(x) has zeros at integers.
            # sinc(a*t) has zeros at t = n/a. Main lobe width = 2/a.
            # Let's stick to scale being the multiplier for now as per previous plan.
            y = func(self.scale * t_shifted)
        elif self.type_name == "Heaviside Step":
            y = func(t_shifted)
        elif self.type_name == "Dirac Delta":
            y = func(t_shifted)
            
        return self.amplitude * y

    def to_latex(self) -> str:
        """
        Return a LaTeX string representation of this component.
        """
        amp_str = f"{self.amplitude}" if self.amplitude != 1 else ""
        if self.amplitude == -1: amp_str = "-"
        
        shift_sign = "-" if self.shift >= 0 else "+"
        shift_val = abs(self.shift)
        shift_str = f"(t {shift_sign} {shift_val})" if self.shift != 0 else "(t)"
        
        if self.type_name == "Rectangular Pulse":
            return f"{amp_str} \\text{{rect}}{shift_str}_{{{self.scale}}}"
        elif self.type_name == "Triangular Pulse":
            return f"{amp_str} \\Lambda{shift_str}_{{{self.scale}}}"
        elif self.type_name == "Sinc Function":
            inner = f"{self.scale}(t {shift_sign} {shift_val})" if self.shift != 0 else f"{self.scale}t"
            if self.scale == 1: inner = shift_str.replace("(", "").replace(")", "")
            return f"{amp_str} \\text{{sinc}}({inner})"
        elif self.type_name == "Heaviside Step":
            return f"{amp_str} u{shift_str}"
        elif self.type_name == "Dirac Delta":
            return f"{amp_str} \\delta{shift_str}"
        return ""

def generate_composite_signal(components: List[SignalComponent], t: np.ndarray) -> np.ndarray:
    """
    Sum all signal components.
    """
    y_total = np.zeros_like(t)
    for comp in components:
        y_total += comp.evaluate(t)
    return y_total
