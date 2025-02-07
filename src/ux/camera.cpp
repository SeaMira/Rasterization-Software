#include "ux/camera.h"

Camera::Camera(int SCR_WIDTH, int SCR_HEIGHT) {
    this->SCR_WIDTH = (float)SCR_WIDTH;
    this->SCR_HEIGHT = (float)SCR_HEIGHT;
    this->lastX = (float)SCR_WIDTH/2.0f;
    this->lastY = (float)SCR_HEIGHT/2.0f;
}

Camera::Camera(int SCR_WIDTH, int SCR_HEIGHT, const glm::vec3 Pos, const glm::vec3 Front, const glm::vec3 Up) {
    this->SCR_WIDTH = SCR_WIDTH;
    this->SCR_HEIGHT = SCR_HEIGHT;
    this->lastX = (float)SCR_WIDTH/2.0f;
    this->lastY = (float)SCR_HEIGHT/2.0f;

    this->cameraPos = Pos;
    this->cameraFront = Front;
    this->cameraUp = Up;
}

glm::vec3 Camera::getPosition() {
    return cameraPos;
}

glm::vec3 Camera::getFront() {
    return cameraFront;
}

glm::vec3 Camera::getUp() {
    return cameraUp;
}

glm::vec3 Camera::getRight() {
    return glm::normalize(glm::cross(cameraFront, cameraUp));
}

glm::mat4 Camera::getProjection() {
    return glm::perspective(glm::radians(fov), SCR_WIDTH/SCR_HEIGHT, 0.1f, 10000.0f);
}

glm::mat4 Camera::getView() {
    return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
}

glm::mat4 Camera::getOrthographic(float left, float right, float bottom, float top, float near, float far) {
    return glm::ortho(left, right, bottom, top, near, far);
}

// hasta que no se necesite ningún cambio solo retornará la identidad
glm::mat4 Camera::getModel() {
    return glm::mat4(1.0f);
}

float Camera::getFov() {
    return fov;
}

float Camera::getPitch() {
    return pitch;
}

float Camera::getYaw() {
    return yaw;
}

void Camera::SetPosition(float x, float y, float z) {
    cameraPos = glm::vec3(x,y,z);
}

void Camera::SetPosition(glm::vec3 newPos) {
    cameraPos = newPos;
}

void Camera::SetFront(float x, float y, float z) {
    cameraFront = glm::vec3(x,y,z);
}

void Camera::SetFront(glm::vec3 newFront) {
    cameraFront = newFront;
}

void Camera::SetUp(float x, float y, float z) {
    cameraUp = glm::vec3(x,y,z);
}

void Camera::SetUp(glm::vec3 newUp) {
    cameraUp = newUp;
}

void Camera::SetMargin(float newMargin) {
    MARGIN = newMargin;
}

void Camera::SetEdgeStep(float newEdgeStep) {
    EDGE_STEP = newEdgeStep;
}

void Camera::SetScrSize(int width, int height) {
    SCR_WIDTH = (float)width; SCR_HEIGHT = (float)height;
}

void Camera::OnKeyboard(int key, float dt) {
    float cameraSpeed = static_cast<float>(mSpeed * dt);
    switch (key) {
        // W
        case 'w':
        {
            cameraPos += cameraSpeed * cameraFront;
        }
        break;
        // S
        case 's':
        {
            cameraPos -= cameraSpeed * cameraFront;
        }
        break;
        // A
        case 'a':
        {
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        }
        break;
        // D
        case 'd':
        {
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        }
        break;
        // SPACE
        case ' ':
        {
            cameraPos +=  initCameraUp* cameraSpeed;
        }
        break;
        // LEFT_SHIFT
        case 'l': 
        {
            cameraPos -=  initCameraUp* cameraSpeed;
        }
        break;
        case 'e': 
        {
            mSpeed +=  1.0f;
        }
        break;
        case 'q': 
        {
            mSpeed -=  1.0f;
            mSpeed = std::max(1.0f, mSpeed);
        }
        break;
    }
}

void Camera::OnMouse(float x, float y) {

    float xpos = x;
    float ypos = y;

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        yaw   = 90.0f;
        pitch =  -45.0f;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f; // change this value to your liking
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw -= xoffset;
    pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    if (abs(xoffset) <= MARGIN && x <= MARGIN) {
        OnLeftEdge = true;
        OnRightEdge = false;
    }
    else if (abs(xoffset) <= MARGIN && x >= (SCR_WIDTH - MARGIN)) {
        OnLeftEdge = false;
        OnRightEdge = true;
    }
    else {
        OnLeftEdge = false;
        OnRightEdge = false;
    }

    if (abs(yoffset) <= MARGIN && y <= MARGIN) {
        OnUpperEdge = true;
        OnLowerEdge = false;
    }
    else if (abs(yoffset) <= MARGIN && y >= (SCR_HEIGHT - MARGIN)) {
        OnUpperEdge = false;
        OnLowerEdge = true;
    }
    else {
        OnUpperEdge = false;
        OnLowerEdge = false;
    }

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.z = sin(glm::radians(pitch));
    // front.y = sin(glm::radians(pitch));
    // front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    glm::vec3 right = glm::normalize(glm::cross(cameraFront, glm::vec3(0.0f, 0.0f, 1.0f))); // Right vector
    cameraUp = glm::normalize(glm::cross(right, cameraFront));
}

void Camera::OnRender(float dt) {
    bool ShouldUpdate = false;

    if (OnLeftEdge) {
        yaw += EDGE_STEP*dt;
        ShouldUpdate = true;
    }
    else if (OnRightEdge) {
        yaw -= EDGE_STEP*dt;
        ShouldUpdate = true;
    }

    if (OnUpperEdge) {
        if (pitch > -90.0f) {
            pitch += EDGE_STEP*dt;
            ShouldUpdate = true;
        }
    }
    else if (OnLowerEdge) {
        if (pitch < 90.0f) {
           pitch -= EDGE_STEP*dt;
           ShouldUpdate = true;
        }
    }

    if (ShouldUpdate) {
        if (pitch > 89.0f)
        pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.z = sin(glm::radians(pitch));
        // front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        // front.y = sin(glm::radians(pitch));
        // front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
        glm::vec3 right = glm::normalize(glm::cross(cameraFront, glm::vec3(0.0f, 0.0f, 1.0f))); // Right vector
        cameraUp = glm::normalize(glm::cross(right, cameraFront));
    }
    
}

void Camera::OnScroll(float yoffset) {
    fov -= yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}