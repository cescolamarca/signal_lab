import numpy as np

def rectpuls(t, w=1):
    """
    Generate a rectangular pulse.
    
    Mimics MATLAB's rectpuls(t, w).
    y = 1 for |t| < w/2
    y = 0.5 for |t| == w/2
    y = 0 for |t| > w/2
    
    Args:
        t: Input time array.
        w: Width of the rectangle. Default is 1.
        
    Returns:
        Numpy array of the signal.
    """
    t = np.asarray(t)
    w = float(w)
    
    # Handle the boundary condition |t| == w/2 carefully with float comparison
    # Using a small epsilon for float comparison if needed, but standard logic first.
    
    y = np.zeros_like(t, dtype=float)
    
    # |t| < w/2
    mask_inside = np.abs(t) < (w / 2.0)
    y[mask_inside] = 1.0
    
    # |t| == w/2
    mask_edge = np.isclose(np.abs(t), (w / 2.0))
    y[mask_edge] = 0.5
    
    return y

def tripuls(t, w=1, s=0):
    """
    Generate a triangular pulse.
    
    Mimics MATLAB's tripuls(t, w, s).
    
    Args:
        t: Input time array.
        w: Width of the triangle. Default is 1.
        s: Skew factor (-1 <= s <= 1). Default is 0 (symmetric).
        
    Returns:
        Numpy array of the signal.
    """
    t = np.asarray(t)
    w = float(w)
    s = float(s)
    
    y = np.zeros_like(t, dtype=float)
    
    # Logic from MATLAB documentation or standard definition
    # Symmetric case (s=0): 1 - |t|/(w/2) for |t| < w/2
    
    # General case:
    # The triangle goes from (t=-w/2, y=0) to (t=s*w/2, y=1) to (t=w/2, y=0)
    
    # Left side: -w/2 <= t < s*w/2
    # Line from (-w/2, 0) to (s*w/2, 1)
    # Slope = 1 / (s*w/2 - (-w/2)) = 1 / (w/2 * (s+1))
    # y = slope * (t - (-w/2)) = (t + w/2) / (w/2 * (s+1))
    
    # Right side: s*w/2 <= t < w/2
    # Line from (s*w/2, 1) to (w/2, 0)
    # Slope = -1 / (w/2 - s*w/2) = -1 / (w/2 * (1-s))
    # y = slope * (t - w/2) = -(t - w/2) / (w/2 * (1-s)) = (w/2 - t) / (w/2 * (1-s))
    
    # Avoid division by zero if s is -1 or 1
    
    if s == 0:
        mask = np.abs(t) < (w / 2.0)
        y[mask] = 1 - np.abs(t[mask]) / (w / 2.0)
        # Edges are 0, so we don't need special handling for |t| == w/2 as it evaluates to 0
    else:
        # Peak location
        t_peak = s * w / 2.0
        
        # Left ramp
        if s > -1:
            mask_left = (t >= -w/2.0) & (t < t_peak)
            y[mask_left] = (t[mask_left] + w/2.0) / ((w/2.0) * (1 + s))
            
        # Right ramp
        if s < 1:
            mask_right = (t >= t_peak) & (t < w/2.0)
            y[mask_right] = (w/2.0 - t[mask_right]) / ((w/2.0) * (1 - s))
            
    # Ensure strictly 0 outside
    y[np.abs(t) >= w/2.0] = 0.0
    
    return y

def sinc(t):
    """
    Generate sinc function.
    
    Mimics MATLAB's sinc(t) = sin(pi*t) / (pi*t).
    Numpy's sinc does exactly this.
    """
    return np.sinc(t)

def heaviside(t):
    """
    Heaviside step function.
    
    Mimics MATLAB's heaviside(t).
    0 for t < 0
    0.5 for t = 0
    1 for t > 0
    """
    return np.heaviside(t, 0.5)

def dirac(t, tolerance=1e-5):
    """
    Approximation of Dirac delta function for numerical plotting.
    
    In MATLAB, dirac(t) is Inf at t=0.
    For plotting purposes, we often represent it as a unit impulse 
    if strictly discrete, or a very high value.
    
    Here, we'll return 1/tolerance if |t| < tolerance (area approx 1 if integrated?),
    OR simpler: return 1.0 at t=0 (discrete impulse) if we assume discrete context,
    OR return infinity.
    
    Given the user asked for "plot di segnali", usually we want to see a spike.
    If we return Inf, matplotlib might not plot it well.
    
    Let's return a "discrete-like" impulse: 1 where t is close to 0, 0 otherwise.
    This is often what's expected in "Signals & Systems" labs unless doing symbolic math.
    """
    y = np.zeros_like(t, dtype=float)
    # Check for values close to 0
    mask = np.isclose(t, 0, atol=1e-10)
    y[mask] = 1.0 # Unit impulse representation
    return y

def get_signal_function(name):
    """
    Retrieve a signal function by its name.
    
    Args:
        name: Name of the signal function (case-insensitive).
        
    Returns:
        The corresponding function or None if not found.
    """
    name = name.lower()
    if name == "rectpuls" or name == "rectangular pulse":
        return rectpuls
    elif name == "tripuls" or name == "triangular pulse":
        return tripuls
    elif name == "sinc" or name == "sinc function":
        return sinc
    elif name == "heaviside" or name == "heaviside step":
        return heaviside
    elif name == "dirac" or name == "dirac delta":
        return dirac
    return None
