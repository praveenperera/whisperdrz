"""gate.py with another run as the reference: gate_vs.py WORKLOAD REF_JSON NEW_JSON

Measures how far NEW moves from REF on gate.py's metrics, e.g. one change against its
own control rather than against the Python reference. The verdict line still prints, but
it isn't the gate
"""
import builtins
import os
import sys

# `python -I` leaves the script's directory off the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402

workload, ref_path, new_path = sys.argv[1:4]
_open = builtins.open
gate.open = lambda p, *a, **k: _open(ref_path if str(p).endswith(".ref.json") else p, *a, **k)
sys.argv = ["gate.py", workload, new_path]
gate.main()
