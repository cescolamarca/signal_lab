import signallab.signals as sig_mod
import sys

print(f"signallab.signals file: {sig_mod.__file__}")
print(f"Attributes of signallab.signals: {dir(sig_mod)}")

if hasattr(sig_mod, 'get_signal_function'):
    print("get_signal_function found!")
else:
    print("get_signal_function NOT found!")
