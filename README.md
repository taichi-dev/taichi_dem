# Taichi DEM
A minimal DEM simulation demo written in Taichi.

<img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_dem/demo.gif" height="270px">

> To implement your own version, click the "Use this template" button on this page and simply modify the `dem.py` script.

## Installation
Make sure your `pip` is up-to-date:

```bash
$ pip3 install pip --upgrade
```

Assume you have a Python 3 environment, to install Taichi:

```bash
$ pip3 install -U taichi
```

To run the demo:

```bash
$ python dem.py
```

## Assumptions
The `dem.py` implements a minimal DEM solver with the following assumptions:

- All paricles are round circles with variable radius.
- Only the normal force between particles is considered - the tangential force is not included.
- The deformation of the particles is not included.
- Ignore the angular momentum of the particle and only consider the translation of the particle.

## Open missions
There are plenty of room for hacking! We suggest a few of them for you to start with:
- Reduce the neighborhood search region from the 3x3 grid cells to only 5 grid cells
- Support more particle geometries
- Implement angular momentum of the particles
- Include tangential forces
- ...

## Show your work!
We encourage you to continue developing on this repo and share your work with our community members. To notify us about your work, make sure you use this repo as a template.
