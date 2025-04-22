# setup.py
from setuptools import setup, find_packages

setup(
    name="pinn-kinetics",  # Name of the package
    version="0.1.0",  # Version of the package
    packages=find_packages(where="src"),  # Look for packages in the src folder
    package_dir={"": "src"},  # Tell setuptools to look in the 'src' folder
    install_requires=[  # List of dependencies for the package
        "torch>=1.8.0",
        "matplotlib>=3.4.0",
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),  # Optional: long description from README
    long_description_content_type="text/markdown",  # Use markdown for long description
)
