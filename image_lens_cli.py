#!/usr/bin/env python3
"""Direct, isolated-Python entry point; independent of the current directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_lens.cli import main
raise SystemExit(main())
