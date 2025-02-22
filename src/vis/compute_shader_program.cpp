#include "vis/compute_shader_program.h"


// constructor generates the shader on the fly
// ------------------------------------------------------------------------
ComputeShader::ComputeShader(const char* computePath)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string computeCode;
    std::ifstream cShaderFile;
    // ensure ifstream objects can throw exceptions:
    cShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        cShaderFile.open(computePath);

        std::stringstream cShaderStream;
        // read file's buffer contents into streams
        cShaderStream << cShaderFile.rdbuf();
        // close file handlers
        cShaderFile.close();
        // convert stream into string
        computeCode = cShaderStream.str();
    }
    catch (std::ifstream::failure& e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
    }
    const char* cShaderCode = computeCode.c_str();
    // 2. compile shaders
    unsigned int compute;
    // compute shader
    compute = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(compute, 1, &cShaderCode, NULL);
    glCompileShader(compute);
    checkCompileErrors(compute, "COMPUTE");
    
    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(compute);
}
void ComputeShader::use() 
{ 
    glUseProgram(ID); 
}

void ComputeShader::setBool(const std::string &name, bool value) const
{         
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); 
}
void ComputeShader::setInt(const std::string &name, int value) const
{ 
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value); 
}
void ComputeShader::setFloat(const std::string &name, float value) const
{ 
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
}
void ComputeShader::setVec2(const std::string &name, const glm::vec2 &value) const
{ 
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ComputeShader::setVec2(const std::string &name, float x, float y) const
{ 
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y); 
}
void ComputeShader::setVec3(const std::string &name, const glm::vec3 &value) const
{ 
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ComputeShader::setVec3(const std::string &name, float x, float y, float z) const
{ 
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z); 
}
void ComputeShader::setVec4(const std::string &name, const glm::vec4 &value) const
{ 
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void ComputeShader::setVec4(const std::string &name, float x, float y, float z, float w) const 
{ 
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w); 
}
void ComputeShader::setVec2I(const std::string &name, const glm::vec2 &value) const
{ 
    glUniform2i(glGetUniformLocation(ID, name.c_str()), value[0], value[1]); 
}
void ComputeShader::setVec2I(const std::string &name, int x, int y) const
{ 
    glUniform2i(glGetUniformLocation(ID, name.c_str()), x, y); 
}
void ComputeShader::setVec3I(const std::string &name, const glm::vec3 &value) const
{ 
    glUniform3i(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[2]); 
}
void ComputeShader::setVec3I(const std::string &name, int x, int y, int z) const
{ 
    glUniform3i(glGetUniformLocation(ID, name.c_str()), x, y, z); 
}
void ComputeShader::setVec4I(const std::string &name, const glm::vec4 &value) const
{ 
    glUniform4i(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[3], value[4]); 
}
void ComputeShader::setVec4I(const std::string &name, int x, int y, int z, int w) 
{ 
    glUniform4i(glGetUniformLocation(ID, name.c_str()), x, y, z, w); 
}
void ComputeShader::setMat2(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ComputeShader::setMat3(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
void ComputeShader::setMat4(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}


void ComputeShader::checkCompileErrors(GLuint shader, std::string type)
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
