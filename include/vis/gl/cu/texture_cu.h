#ifndef _TEXTURE_CU_H_
#define _TEXTURE_CU_H_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "vis/gl/texture.h"
#include "utils/parallel/cuda_checks.h"

/**
 * @class TextureCUDAWrapper
 * 
 * @brief TextureCUDAWrapper class wrapper.
 * 
 * Texture class wrapper for using textures on shaders. Used by Canvas objects since a texture works as viewport.
 */
class TextureCUDAWrapper {
public:

    TextureCUDAWrapper();
    TextureCUDAWrapper(Texture& texture, GLenum target, unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);
    ~TextureCUDAWrapper();

    void setup(Texture& texture, GLenum target, unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);
    void cudaRegisterTex(unsigned int flags=cudaGraphicsRegisterFlagsSurfaceLoadStore);

    Texture* getTextureRef() const;
    void setTexture(Texture& texture);
    GLuint getTextureId() const;

    void setTarget(GLenum target);
    GLenum getTarget() const;

    void cudaMapResources();
    void cudaCreateSurfaceObj();
    void cudaDestroySurfaceObj();
    void cudaUnmapResources();
    void cudaUnregisterTex();
    cudaSurfaceObject_t getSurfaceObject() const;

private:

    GLenum m_target; ///< target texture to set.
    Texture* m_texture; ///< Texture object.

    cudaGraphicsResource* m_cudaResource;
    cudaArray* m_textureArray;
    cudaResourceDesc m_resDesc = {};
    cudaSurfaceObject_t m_surfaceObj = 0;

    bool m_mapped = false;
};

#endif