"""Setup script for the disease-gene prediction research repository."""

from setuptools import find_packages, setup


setup(
    name="interpretable-disease-gene-prediction",
    version="0.1.0",
    description="Hybrid graph learning and metapath reasoning for interpretable disease-gene prediction",
    author="Research Engineering Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "networkx>=3.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "tqdm>=4.66",
        "torch>=2.2",
        "torch-geometric>=2.5",
    ],
)
