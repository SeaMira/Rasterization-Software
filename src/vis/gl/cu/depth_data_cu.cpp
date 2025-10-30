#include "vis/gl/cu/depth_data_cu.h"


DepthDataCUDA::DepthDataCUDA(int width, int height) :
    m_screenHeight(height),
    m_screenWidth(width),
    m_depthBuffer(nullptr)
{
    memoryAlloc();
}

DepthDataCUDA::~DepthDataCUDA() 
{
    cudaFree(m_depthBuffer);
}

void DepthDataCUDA::setup(int width, int height)
{
    m_screenHeight = height;
    m_screenWidth = width;
    memoryAlloc();
}


void DepthDataCUDA::resolutionSetup(int width, int height)
{
    m_screenHeight = height;
    m_screenWidth = width;
}

void DepthDataCUDA::memoryAlloc()
{
    cudaMalloc(&m_depthBuffer, m_screenWidth * m_screenHeight * sizeof(unsigned int));
    cudaMemset(m_depthBuffer, 0, m_screenWidth * m_screenHeight * sizeof(unsigned int));
}

unsigned int* DepthDataCUDA::getDepthBuffer() const 
{ 
    return m_depthBuffer; 
}

int DepthDataCUDA::getScreenWidth() const 
{ 
    return m_screenWidth; 
}

int DepthDataCUDA::getScreenHeight() const 
{ 
    return m_screenHeight; 
}
