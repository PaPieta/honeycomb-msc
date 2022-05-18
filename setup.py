from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="honeycomb",
    version="0.1.0",
    author="Pawel Pieta",
    author_email="niejep@dtu.dk",
    description="A Python package for segmentation and model generation from CT scans of folded honeycombs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaPieta/honeycomb-msc",
    packages=["honeycomb"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=["numpy", "scipy", "scikit-learn", "slgbuilder", "shrdr", "scikit-image", "matplotlib", "opencv-python"],
)