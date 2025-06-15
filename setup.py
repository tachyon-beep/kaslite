# setup.py
from setuptools import setup, find_packages

setup(
    name="morphogenetic_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn"
    ],
)

