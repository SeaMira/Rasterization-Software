# Rasterization-Software


Project dedicated on avoid the utilization of the hardware pipeline for the creation of real time visualization of simple scenes with a massive amount of elements.

It's intended to be used in molecular visualization, therefore atoms will be represented as spheres and bonds as cylinders.

Sphere and cylinders will be proyected extracting their 2D bounding box and then casting rays through them to get the depth and normal on each pixel.

Frustum culling using general approach: each entity is tested to be in front or over the six planes of the camera frustum.

Occlusion culling using hierarchical z-buffer by building mipmap leves generalizing areas depth values. These mipmaps are used to test against the newer depth values to see if the entities are occluded or not.

Molecular visualization is achieved by extracting scenes from molecular files.

## What is used for this?

It uses OpenGL as the graphics API and SDL3 for user interface/input settings. Since we avoid the use of standard pipeline, there are no shaders except compute shaders, for massive parallel computing.

Other tools may be:
* GLM: mathematical operations.
* GLAD: linking to GL functions.
* chemfiles: chemistry file loader.
* OpenMP: for CPU parallelization, drawing entities acceleration.
* ImGui: UI for showing scene, benchmar, camera and general info.

## Methods Implemented

### Sequential Version

Helps for debugging and trying new algorithms, with easier control on every step.

The Hierarchical z-buffer is built every frame, every mipmap level at a time.

### Parallel Versions

For hierarchical z-buffer construction it uses a one compute shader pass, building only the desired mipmap level. 

#### First Simple version: assign an entity to each thread (one-thread-one-entity)

First GPU version that makes every thread to draw an entity. Each entity uses a SSBO to store its info, and another SSBO to store depths. May have data-racing situations when depth has to be tested and image has to be overwritten so atomic operations were used.

#### Second version: extract proyection info. (shader one: one-thread-one-enity) and bbox drawing (shader two: one-thread-one-pixel)

Second GPU version which extracts the bounding box and projection info. of entities on first shader, stores it on a buffer and sends it to a second shader, in which every thread checks intersections of its pixel with the bounding boxes and takes care of shading.

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

## VS Solution

It has 5 main projects:
* cylinder_projection_cpu : CPU implementation. It shows scenes with spheres and cylinders, with or without occlusion culling.

* fst_cpu_w_cylinders : GPU implementation with first version. Uses CPU frustum culling of entities with or withour occlusion culling.

* fst_gpu_w_cylinders : GPU implementation with first version. Uses GPU frustum culling of entities with or withour occlusion culling.

* snd_cpu_w_cylinders : GPU implementation with second version. Uses CPU frustum culling of entities with or withour occlusion culling.

* snd_gpu_w_cylinders : GPU implementation with second version. Uses GPU frustum culling of entities with or withour occlusion culling.