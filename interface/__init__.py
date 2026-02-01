"""
FMI Depth Interface - Core Mathematical Interface

A deterministic, parameter-free interface for depth-stratified systems.
"""

from .phi import PhiCoordinate
from .ladder import LadderClosure
from .response import ResponseFunctional
from .determinism import HashValidator

__version__ = "0.1.0-fmi"
__all__ = ["PhiCoordinate", "LadderClosure", "ResponseFunctional", "HashValidator"]
