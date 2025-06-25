from setuptools import setup, find_packages

setup(
    name="pyautoencoder",
    version="1.0.0",
    description="A Python package offering implementations of state-of-the-art autoencoder architectures in PyTorch.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Andrea Pollastro",
    url="https://github.com/andrea-pollastro/pyautoencoder",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0"
    ],
    keywords=[
        "autoencoder", "vae", "pytorch", "deep learning",
        "machine learning", "representation learning", "dimensionality reduction",
        "generative models"
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7',
)
