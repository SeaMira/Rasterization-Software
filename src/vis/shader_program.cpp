#include "vis/shader_program.h"


ShaderProgram::ShaderProgram(const char* vertexPath, const char* fragmentPath, const char* geometryPath, const char* shaderProgramName,
const char* vertexShaderName, const char* fragmentShaderName, const char* geometryShaderName)
{
    m_shaderProgramName = shaderProgramName;
    ID = glCreateProgram();
    glObjectLabel(GL_PROGRAM, ID, -1, m_shaderProgramName.c_str());
    // vertex shader
    if (vertexPath != "")
        attachShader(vertexPath, GL_VERTEX_SHADER, vertexShaderName);
    // fragment shader
    if (fragmentPath != "")
        attachShader(fragmentPath, GL_FRAGMENT_SHADER, fragmentShaderName);
    // if geometry shader
    if (geometryPath != "")
        attachShader(geometryPath, GL_GEOMETRY_SHADER, geometryShaderName);
}

void ShaderProgram::attachShader(const char* shaderPath, GLenum shaderType, const char* shaderName)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string shaderCode;
    std::ifstream sShaderFile;
    // ensure ifstream objects can throw exceptions:
    sShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        sShaderFile.open(shaderPath);
        std::stringstream sShaderStream;
        // read file's buffer contents into streams
        sShaderStream << sShaderFile.rdbuf();
        // close file handlers
        sShaderFile.close();
        // convert stream into string
        shaderCode = sShaderStream.str();
    }
    catch (std::ifstream::failure& e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
    }
    const char* sShaderCode = shaderCode.c_str();
    // 2. compile shaders
    unsigned int shader;
    // vertex shader
    shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &sShaderCode, NULL);
    glCompileShader(shader);
    checkCompileErrors(shader, "SHADER");
    
    glAttachShader(ID, shader);

    glObjectLabel(GL_SHADER, shader, -1, shaderName);
    
    glDeleteShader(shader);
}


void ShaderProgram::linkProgram()
{
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
}

void ShaderProgram::use() 
{ 
    glUseProgram(ID); 
}

void ShaderProgram::setBool(const std::string &name, bool value) const
{         
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); 
}
void ShaderProgram::setInt(const std::string &name, int value) const
{ 
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value); 
}
void ShaderProgram::setUint(const std::string &name, unsigned int value) const
{ 
    glUniform1ui(glGetUniformLocation(ID, name.c_str()), value); 
}
void ShaderProgram::setFloat(const std::string &name, float value) const
{ 
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
}
void ShaderProgram::setVec2(const std::string &name, const glm::vec2 &value) const
{ 
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ShaderProgram::setVec2(const std::string &name, float x, float y) const
{ 
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y); 
}
void ShaderProgram::setVec3(const std::string &name, const glm::vec3 &value) const
{ 
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ShaderProgram::setVec3(const std::string &name, float x, float y, float z) const
{ 
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z); 
}
void ShaderProgram::setVec4(const std::string &name, const glm::vec4 &value) const
{ 
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ShaderProgram::setVec4(const std::string &name, float x, float y, float z, float w) const 
{ 
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w); 
}
void ShaderProgram::setVec2I(const std::string &name, const glm::vec2 &value) const
{ 
    glUniform2i(glGetUniformLocation(ID, name.c_str()), value[0], value[1]); 
}
void ShaderProgram::setVec2I(const std::string &name, int x, int y) const
{ 
    glUniform2i(glGetUniformLocation(ID, name.c_str()), x, y); 
}
void ShaderProgram::setVec3I(const std::string &name, const glm::vec3 &value) const
{ 
    glUniform3i(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[2]); 
}
void ShaderProgram::setVec3I(const std::string &name, int x, int y, int z) const
{ 
    glUniform3i(glGetUniformLocation(ID, name.c_str()), x, y, z); 
}
void ShaderProgram::setVec4I(const std::string &name, const glm::vec4 &value) const
{ 
    glUniform4i(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[3], value[4]); 
}
void ShaderProgram::setVec4I(const std::string &name, int x, int y, int z, int w) 
{ 
    glUniform4i(glGetUniformLocation(ID, name.c_str()), x, y, z, w); 
}
void ShaderProgram::setMat2(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setMat3(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
void ShaderProgram::setMat4(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::checkCompileErrors(GLuint shader, std::string type)
{
    GLint success;
    GLchar infoLog[1024];
    if(type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if(!success)
        {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}