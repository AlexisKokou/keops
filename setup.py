from setuptools import setup, find_packages

setup(
    name="keops_jax",
    version="0.1.0",
    packages=['keops_jax'],
    package_dir={'keops_jax': '.'},  # IMPORTANT: dit que keops_jax est ici
    install_requires=[
        "jax",
        "jaxlib",
        "pykeops>=2.2.0",
        "numpy",
    ],
)
