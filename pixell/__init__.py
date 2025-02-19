# -*- coding: utf-8 -*-

"""Top-level package for pixell."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pixell")
except PackageNotFoundError:
    __version__ = "unknown"
    
__author__ = """Simons Observatory Collaboration Analysis Library Task Force"""
__email__ = ''
