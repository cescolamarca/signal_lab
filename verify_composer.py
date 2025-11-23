import numpy as np
from signallab.composer import SignalComponent, generate_composite_signal

def test_composer_logic():
    t = np.linspace(-5, 5, 100)
    
    # Component 1: Rect pulse, width 2, centered at 0
    c1 = SignalComponent(type_name="Rectangular Pulse", scale=2.0)
    y1 = c1.evaluate(t)
    
    # Check center value (t=0) should be 1
    idx_0 = np.argmin(np.abs(t))
    assert np.isclose(y1[idx_0], 1.0), f"Rect center failed: {y1[idx_0]}"
    
    # Component 2: Tri pulse, width 2, shifted by 2
    c2 = SignalComponent(type_name="Triangular Pulse", scale=2.0, shift=2.0)
    y2 = c2.evaluate(t)
    
    # Check peak at t=2
    idx_2 = np.argmin(np.abs(t - 2.0))
    assert np.isclose(y2[idx_2], 1.0), f"Tri peak failed: {y2[idx_2]}"
    
    # Composite
    y_total = generate_composite_signal([c1, c2], t)
    
    # At t=0, y_total should be ~1 (rect) + 0 (tri is far away)
    assert np.isclose(y_total[idx_0], 1.0, atol=0.1), f"Composite at t=0 failed: {y_total[idx_0]}"
    
    # At t=2, y_total should be ~0 (rect is far away) + 1 (tri)
    assert np.isclose(y_total[idx_2], 1.0, atol=0.1), f"Composite at t=2 failed: {y_total[idx_2]}"
    
    print("Composer logic verification passed!")

if __name__ == "__main__":
    test_composer_logic()
