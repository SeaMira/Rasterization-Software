# Rasterization-Software


Project dedicated on avoid the utilization of the hardware pipeline for the creation of real time visualization of simple scenes with a massive amount of elements.

It's intended to be used in molecular visualization and with that in mind the first bodies to rasterize by hand are spheres.

## What is used for this?

It uses OpenGL as the graphics API and SDL3 for user interface/input settings. Since we avoid the use of standard pipeline, there are no shaders except compute shaders, for massive parallel computing.

Other tools may be:
* GLM: mathematical operations.
* GLAD: linking to gl functions.
* chemfiles: chemistry file loader

## Methods Implemented

### Sequential Version

Spheres are projected as impostors: bounding box is used to test casted rays through it and draw (with a simple gouraud shading) and iluminated pixel.

### Parallel Versions

#### First Simple version: assign a sphere to each thread

Simple initial version that makes every thread to draw a sphere. Uses a SSBO to store sphere positions and another SSBO to store depths. May have data-racing situations when depth has to be tested and image has to be overwritten.


## How to compile?

First clone or download the code of this project:

```
git clone https://github.com/SeaMira/Rasterization-Software.git
```

Then in terminal located at the repo use the following commands for :
```
git submodule update --init --recursive
```

This will upload the external repositories used on the project.

To compile a release version installable with ninja use:
```
cmake --preset ninja-release
```

To compile a visual studio solution version use:
```
cmake --preset vs-debug
```
