from setuptools import setup, find_packages

setup(
    name="pyautoencoder",
    version="1.0.0",
    description="PyTorch implementations of state-of-the-art autoencoder architectures.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Andrea Pollastro",
    url="https://github.com/andrea-pollastro/pyautoencoder",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
