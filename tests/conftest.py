import os
import sys

# All project modules live in src/ (flat); make them importable in tests.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
