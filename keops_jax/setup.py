# keops_jax/setup.py
from setuptools import setup, find_packages

setup(
    name="keops_jax",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "pykeops>=2.0.0",
    ],
    author="Votre Nom",
    description="JAX bindings for KeOps kernel operations",
    python_requires=">=3.8",
)