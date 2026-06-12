"""Pytest configuration — adds project root to sys.path."""
import sys
from pathlib import Path

# Ensure project root is on path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent))