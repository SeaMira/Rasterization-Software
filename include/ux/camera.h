#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>
#include <iostream>
#include "glm/common.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include <glm/gtc/quaternion.hpp>
#include "glm/gtc/type_ptr.hpp"

class Camera {
public:

    Camera(int SCR_WIDTH, int SCR_HEIGHT);

    Camera(int SCR_WIDTH, int SCR_HEIGHT, const glm::vec3 Pos, const glm::vec3 Front, const glm::vec3 Up);

    void SetPosition(float x, float y, float z);
    void SetPosition(glm::vec3 newPos);
    void SetFront(float x, float y, float z);
    void SetFront(glm::vec3 newPos);
    void SetUp(float x, float y, float z);
    void SetUp(glm::vec3 newPos);
    void SetMargin(float newMargin);
    void SetEdgeStep(float newEdgeStep);
    void SetScrSize(int width, int height);
    
    // void updateCameraVectors();
    void OnKeyboard(int key, float dt);
    void OnMouse(float x, float y);
    void OnRender(float dt);
    void OnScroll(float yoffset);

    glm::vec3 getPosition();
    glm::vec3 getFront();
    glm::vec3 getUp();
    glm::vec3 getRight();

    glm::mat4 getProjection();
    glm::mat4 getOrthographic(float left, float right, float bottom, float top, float near, float far);
    glm::mat4 getView();
    glm::mat4 getModel();
    float getFov();
    float getYaw();
    float getPitch();


    bool firstMouse = true;
    float yaw   = 90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
    float pitch =  -45.0f;
    float lastX;
    float lastY;
    float fov   =  45.0f;
    float SCR_WIDTH;
    float SCR_HEIGHT;

    float MARGIN = 20.0f;
    float EDGE_STEP = 70.0f;


private:
    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  20.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 0.0f,  1.0f);
    glm::vec3 initCameraUp    = glm::vec3(0.0f, 0.0f,  1.0f);

    bool OnUpperEdge;
    bool OnLowerEdge;
    bool OnLeftEdge;
    bool OnRightEdge;

    float mSpeed = 10.0f;
    
};

#endif // _CAMERA_H_