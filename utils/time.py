"""
EP-Neural-Nets: Time Formatting Utilities
This module provides simple helper functions for converting seconds into 
human-readable (hh:mm:ss) formats, primarily used to output estimated 
remaining times during parameter sweeps.
"""

def hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"