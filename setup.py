import os
import sys
import re
import platform
import subprocess
from setuptools import setup, Extension, find_packages

setup(
    name="torchmodels",
    version="1.0.0",
    author="Jørgen Henriksen",
    description="Commonly used neural network architectures implemented using PyTorch",
    long_description="",
    packages=["torchmodels"],
    zip_safe=False,
)
