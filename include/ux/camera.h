#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>
#include <iostream>
#include "glm/common.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include <glm/gtc/quaternion.hpp>
#include "glm/gtc/type_ptr.hpp"


/**
 * @class Camera
 * 
 * @brief Camera class for setting camera in the scene.
 * 
 * The Camera class is used to control the camera in the scene, it has the position, front, up and right vectors
 * and the view and projection matrices to be used in the rendering process.
 */
class Camera {
public:
    /**
     * @brief Constructor for the Camera class.
     * 
     * Constructor for the Camera class, initializes the camera with the screen width and height, the vectors initialize 
     * with default values.
     * 
     * @param SCR_WIDTH the width of the screen.
     * @param SCR_HEIGHT the height of the screen.
     */
    Camera(int SCR_WIDTH, int SCR_HEIGHT);


    /**
     * @brief Constructor for the Camera class.
     * 
     * Constructor for the Camera class, initializes the camera with the screen width and height, the position, front and up vectors.
     * 
     * @param SCR_WIDTH the width of the screen.
     * @param SCR_HEIGHT the height of the screen.
     * @param Pos the position of the camera.
     * @param Front the front vector of the camera.
     * @param Up the up vector of the camera.
     */
    Camera(int SCR_WIDTH, int SCR_HEIGHT, const glm::vec3 Pos, const glm::vec3 Front, const glm::vec3 Up);

    /**
     * @brief Set the position of the camera.
     * 
     * @param x the x component of the new position.
     * @param y the y component of the new position.
     * @param z the z component of the new position.
     */
    void SetPosition(float x, float y, float z);

    /**
     * @brief Set the position of the camera.
     * 
     * @param newPos the new position of the camera.
     */	
    void SetPosition(glm::vec3 newPos);

    /**
     * @brief Set the front vector of the camera.
     * 
     * @param x the x component of the new front vector.
     * @param y the y component of the new front vector.
     * @param z the z component of the new front vector.
     */
    void SetFront(float x, float y, float z);

    /**
     * @brief Set the front vector of the camera.
     * 
     * @param newPos the new front vector of the camera.
     */
    void SetFront(glm::vec3 newPos);

    /**
     * @brief Set the up vector of the camera.
     * 
     * @param x the x component of the new up vector.
     * @param y the y component of the new up vector.
     * @param z the z component of the new up vector.
     */
    void SetUp(float x, float y, float z);

    /**
     * @brief Set the up vector of the camera.
     * 
     * @param newPos the new up vector of the camera.
     */
    void SetUp(glm::vec3 newPos);

    /**
     * @brief Set the margin of the camera.
     * 
     * Sets the trigger margin for the camera to rotate.
     */
    void SetMargin(float newMargin);

    /**
     * @brief Set the edge step of the camera.
     * 
     * Sets the movement step of the camera when mouse is on the margin.
     * 
     * @param newEdgeStep the new step of the camera.
     */
    void SetEdgeStep(float newEdgeStep);

    /**
     * @brief Set the screen size of the camera.
     * 
     * Sets the screen size of the camera.
     * 
     * @param width the width of the screen.
     * @param height the height of the screen.
     */
    void SetScrSize(int width, int height);

    /**
     * @brief Look at a target.
     * 
     * Looks at a target in the scene or at a certain direction.
     * 
     * @param target the target to look at.
     */
    void lookAtTarget(const glm::vec3& target);
    
    /**
     * @brief OnKeyboard event.
     * 
     * Handles the keyboard events for the camera on certain keys.
     * 
     * @param key the key pressed to check.
     * @param dt the delta time of the frame.
     */
    void OnKeyboard(int key, float dt);

    /**
     * @brief OnMouse event.
     * 
     * Handles the camera settings on mouse movement.
     * 
     * @param x the x position of the mouse.
     * @param y the y position of the mouse.
     */
    void OnMouse(float x, float y);

    /**
     * @brief OnRender event.
     * 
     * Handles the camera settings on render. Needs to adapt depending on the mouse
     * position (on margin or not).
     * 
     * @param dt the delta time of the frame.
     */
    void OnRender(float dt);

    /**
     * @brief OnScroll event.
     * 
     * Handles the camera settings on scroll. Changes the field of view of the camera.
     * Up scroll increases the field of view, down scroll decreases it.
     * 
     * @param yoffset the offset of the scroll.
     */
    void OnScroll(float yoffset);

    /**
     * @brief Get the position of the camera.
     * 
     * @return the position of the camera.
     */
    glm::vec3 getPosition();

    /**
     * @brief Get the front vector of the camera.
     * 
     * @return the front vector of the camera.
     */
    glm::vec3 getFront();

    /**
     * @brief Get the up vector of the camera.
     * 
     * @return the up vector of the camera.
     */
    glm::vec3 getUp();

    /**
     * @brief Get the right vector of the camera.
     * 
     * @return the right vector of the camera.
     */
    glm::vec3 getRight();

    /**
     * @brief Get the projection matrix of the camera.
     * 
     * @return the projection matrix of the camera.
     */
    glm::mat4 getProjection();

    /**
     * @brief Get the orthographic matrix of the camera.
     * 
     * @param left the left coordinate of the orthographic matrix.
     * @param right the right coordinate of the orthographic matrix.
     * @param bottom the bottom coordinate of the orthographic matrix.
     * @param top the top coordinate of the orthographic matrix.
     * @param near the near coordinate of the orthographic matrix.
     * @param far the far coordinate of the orthographic matrix.
     * 
     * @return the orthographic matrix of the camera.
     */
    glm::mat4 getOrthographic(float left, float right, float bottom, float top, float near, float far);
    
    /**
     * @brief Get the view matrix of the camera.
     * 
     * @return the view matrix of the camera.
     */
    glm::mat4 getView();

    /**
     * @brief Get the model matrix of the camera.
     * 
     * @return the model matrix of the camera.
     */
    glm::mat4 getModel();

    /**
     * @brief Get the field of view of the camera.
     * 
     * @return the field of view of the camera.
     */
    float getFov();

    /**
     * @brief Get the yaw of the camera.
     * 
     * Get the yaw of the camera. The yaw is the angle of the camera in the xz plane.
     * 
     * @return the yaw of the camera.
     */
    float getYaw();

    /**
     * @brief Get the pitch of the camera.
     * 
     * Get the pitch of the camera. The pitch is the angle of the camera in the yz plane.
     * 
     * @return the pitch of the camera.
     */
    float getPitch();

    /**
     * @brief Get the far plane distance of the camera.
     * 
     * @return the far plane distance of the camera.
     */
    float getFar();

    /**
     * @brief Get the near plane distance of the camera.
     * 
     * @return the near plane distance of the camera.
     */
    float getNear();

    /**
     * @brief Updates the camera vectors on rotation, fov, yaw and pitch changes
     */
    void update();

    /**
     * @brief Show the camera information.
     * 
     * Simple on-terminal camera information print.
     */
    void showInfo();


    bool firstMouse = true; ///< whether the mouse is on the screen for the first time.
    float yaw   = 180.f;	///< xz angle of mouse. Default initialization on 180°. 
    float pitch =  0.f; ///< yz angle of mouse. Default initialization on 0°.
    float lastX; ///< last x position of the mouse.
    float lastY; ///< last y position of the mouse;
    float fov   =  45.0f; ///< field of view of the camera. Default initialization on 45°.
    float SCR_WIDTH; ///< width of the screen.
    float SCR_HEIGHT; ///< height of the screen.

    float MARGIN = 20.0f; ///< margin of the screen for the camera to rotate.
    float EDGE_STEP = 70.0f; ///< step of the camera when on the margin.


private:
    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  0.0f); ///< position of the camera.
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 1.0f); ///< front vector of the camera.
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f); ///< up vector of the camera.
    glm::vec3 cameraRight    = glm::vec3(-1.0f, 0.0f,  0.0f); ///< right vector of the camera.
    glm::vec3 initCameraUp    = glm::vec3(0.0f, 1.0f,  0.0f); ///< initial up vector of the camera.

    bool OnUpperEdge; ///< whether the mouse is on the upper edge of the screen.
    bool OnLowerEdge; ///< whether the mouse is on the lower edge of the screen.
    bool OnLeftEdge; ///< whether the mouse is on the left edge of the screen.
    bool OnRightEdge; ///< whether the mouse is on the right edge of the screen.

    float mSpeed = 10.0f; ///< speed of the camera.
    float mNear = 0.1f;  ///< near plane distance of the camera.
    float mFar = 100.0f; ///< far plane distance of the camera.

    friend class CameraInfoComponent; ///< the camera info component is a friend class so it can be shown on UI.
    
};

#endif // _CAMERA_H_