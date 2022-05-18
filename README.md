# honeycomb-msc
> Pyhton library for segmentation and FEM model generation of folded honeycomb structures from volumetric images obtained using a CT scanner.

The project holds a set of functions and classes that allow for segmentation of folded honeycomb scans by unfolding each wall and detecting the edge surfaces through solving Optimal Net Surface Detection problem.

The library was developed as a part of Master Thesis project at Technical University of Denmark (DTU), under supervision of [Vedrana Andersen Dahl](https://orbit.dtu.dk/en/persons/vedrana-andersen-dahl) and [Lars Pilgaard Mikkelsen](https://orbit.dtu.dk/en/persons/lars-pilgaard-mikkelsen). Details of the project can be found inside the thesis: "3D surface fitting for segmenting and modelling folded honeycombs".

The segmentation is powered by [slgbuilder](https://github.com/Skielex/slgbuilder)  library and its extension [shrdr](https://github.com/Skielex/shrdr). 

Other external libraries:
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/), 
* [SciPy](https://scipy.org/), 
* [scikit-learn](https://scikit-learn.org/stable/), 
* [scikit-image](https://scikit-image.org/), 
* [OpenCV](https://opencv.org/).

# Getting started

## Installation

Clone the repository:
* HTTPS
```sh
   https://github.com/PaPieta/honeycomb-msc.git
```
* SSH
```sh
   git@github.com:PaPieta/honeycomb-msc.git
```
Install the library:
```sh
   cd honeycomb-msc
   pip install .
```

# Prerequisites
All required packages, except for *shrdr* can be installed via provided requirements file:
```sh
  pip install -r requirements.txt
```
To install shrdr, go to https://github.com/Skielex/shrdr and follow the installation instructions.


## Examples

Example use of the library is presented in a form of notebooks on Code Ocean platform (To be updated)