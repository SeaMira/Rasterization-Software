#include "utils/parallel/aux_functions.h"

void cullSpheres(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount, int indexOffset)
{
    visibleSpheresCount = 0;
    for (int i = 0 ; i < spheres.size(); i++)
        if (frustum.isSphereInside(spheres[i]))
            visibleSpheres[visibleSpheresCount++] = {spheres[i], i + indexOffset, {0, 0, 0}};    
}

void cullSimpleSpheres(std::vector<Sphere>& spheres, std::vector<Sphere>& visibleSpheres, Frustum& frustum, int& visibleSpheresCount, int indexOffset)
{
    visibleSpheresCount = 0;
    for (int i = 0 ; i < spheres.size(); i++)
        if (frustum.isSphereInside(spheres[i]))
            visibleSpheres[visibleSpheresCount++] = spheres[i];    
}

void fillSpheresData(std::vector<Sphere>& spheres, std::vector<SphereContainer>& visibleSpheres, int indexOffset)
{
    for (int i = 0 ; i < spheres.size(); i++)
        visibleSpheres[i] = {spheres[i], i + indexOffset, {0, 0, 0}};    
}


void cullSimpleCylinders(std::vector<Cylinder>& cylinders, std::vector<Cylinder>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount, int indexOffset)
{
    visibleCylindersCount = 0;
    for (int i = 0 ; i < cylinders.size(); i++)
        if (frustum.isCylinderInside(cylinders[i]))
            visibleCylinders[visibleCylindersCount++] = cylinders[i];    
}

void cullCylinders(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, Frustum& frustum, int& visibleCylindersCount, int indexOffset)
{
    visibleCylindersCount = 0;
    for (int i = 0 ; i < cylinders.size(); i++)
        if (frustum.isCylinderInside(cylinders[i]))
        visibleCylinders[visibleCylindersCount++] = {
            cylinders[i].pa_r, 
            cylinders[i].pb_r,
            i + indexOffset, 
            {0, 0, 0}
        };    
}

void fillCylindersData(std::vector<Cylinder>& cylinders, std::vector<CylinderContainer>& visibleCylinders, int indexOffset)
{
    for (int i = 0 ; i < cylinders.size(); i++)
        visibleCylinders[i] = {
            cylinders[i].pa_r, 
            cylinders[i].pb_r,
            i + indexOffset, 
            {0, 0, 0}
        };  
}

void setCameraUniforms(ComputeShader& shader, Camera& camera)
{
    shader.setMat4("proj", camera.getProjection());
    shader.setMat4("view", camera.getView());
    shader.setVec3("up", camera.getUp());
    shader.setVec3("front", camera.getFront());
    shader.setVec3("right", camera.getRight());
    shader.setVec3("cameraPos", camera.getPosition());
    shader.setFloat("fov", camera.getFov());
}

void setCameraUniforms(ShaderProgram& shader, Camera& camera)
{
    shader.setMat4("proj", camera.getProjection());
    shader.setMat4("view", camera.getView());
    shader.setVec3("up", camera.getUp());
    shader.setVec3("front", camera.getFront());
    shader.setVec3("right", camera.getRight());
    shader.setVec3("cameraPos", camera.getPosition());
    shader.setFloat("fov", camera.getFov());
}

void setFrustumUniforms(ComputeShader& shader, Frustum& frustum)
{
    shader.setVec4("frustumTopFace", glm::vec4(frustum.topFace.normal, frustum.topFace.distance));
    shader.setVec4("frustumBottomFace", glm::vec4(frustum.bottomFace.normal, frustum.bottomFace.distance));
    shader.setVec4("frustumRightFace", glm::vec4(frustum.rightFace.normal, frustum.rightFace.distance));
    shader.setVec4("frustumLeftFace", glm::vec4(frustum.leftFace.normal, frustum.leftFace.distance));
    shader.setVec4("frustumFarFace", glm::vec4(frustum.farFace.normal, frustum.farFace.distance));
    shader.setVec4("frustumNearFace", glm::vec4(frustum.nearFace.normal, frustum.nearFace.distance));
}

void dispatchComputeShaderWithLabel(GLuint numGroupsX, GLuint numGroupsY, const std::string& label, GLbitfield barrier)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label.c_str());
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    if (barrier != 0)
        glMemoryBarrier(barrier);
    glPopDebugGroup();
}