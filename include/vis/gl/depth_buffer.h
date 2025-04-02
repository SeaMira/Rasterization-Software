#ifndef _DEPTH_BUFFER_
#define _DEPTH_BUFFER_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include "vis/gl/texture.h"

class DepthBuffer
{
public:

    DepthBuffer(int width, int height, int bindingPoint);
    ~DepthBuffer();

    void bind();
    void bindImage(GLuint unit, GLenum access);
    void unbind();

private:
    int m_bindingPoint;
    GLuint m_depthSampler;
    Texture m_texture;
};

#endif // _DEPTH_BUFFER_