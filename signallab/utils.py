"""
Utility functions for SignalLab.

This module provides helper functions for number formatting and safe mathematical
expression parsing for signal generation.
"""

import numpy as np

def format_number(value, decimals=2):
    """
    Format a number to a string with a fixed number of decimals.
    Removes trailing zeros if they are not significant, but respects the max decimals.
    
    Args:
        value: The number to format.
        decimals: Maximum number of decimals to show.
        
    Returns:
        Formatted string.
    """
    if value is None:
        return ""
    
    # Handle numpy types
    if hasattr(value, 'item'):
        value = value.item()
        
    if isinstance(value, (int, float)):
        # Format with fixed precision
        s = f"{value:.{decimals}f}"
        # Remove trailing zeros and decimal point if integer
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return str(value)

def parse_expression(expression, t):
    """
    Safe evaluation of a mathematical expression string.
    
    Args:
        expression: String expression (e.g., "sin(2*pi*t)").
        t: Time array.
        
    Returns:
        Signal array.
    """
    # Allowed context
    context = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "t": t,
        "np": np
    }
    
    try:
        # Evaluate expression
        # limiting builtins to None for safety
        return eval(expression, {"__builtins__": None}, context)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")
