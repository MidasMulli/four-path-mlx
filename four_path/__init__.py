"""
Four-Path Speculative Decoding for Apple Silicon
=================================================

Four prediction sources, three processors, one generate loop.

  CPU  N-gram hash   → pattern-based draft chains
  ANE  1.7B CoreML   → neural lookahead draft tokens
  MTP  GPU head      → hidden-state prediction
  GPU  9B backbone   → verification + generation

Usage:
    from four_path.generate import FourPathDrafter, four_path_generate_step
    from four_path.ngram import NgramPredictor
    from four_path.mtp_patch import patch_mtp
"""

from four_path.generate import FourPathDrafter, four_path_generate_step
from four_path.ngram import NgramPredictor
