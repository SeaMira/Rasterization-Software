#include "core/shader/shader_base.h"
#include <unordered_map>

ShaderBase::ShaderBase(const std::string& name)
    : m_name(name)
{
}

ShaderBase::~ShaderBase()
{
    if (m_programId != 0)
    {
        glDeleteProgram(m_programId);
        m_programId = 0;
    }
}

ShaderBase::ShaderBase(ShaderBase&& other) noexcept
    : m_programId(other.m_programId)
    , m_name(std::move(other.m_name))
    , m_uniformCache(std::move(other.m_uniformCache))
{
    other.m_programId = 0;
}

ShaderBase& ShaderBase::operator=(ShaderBase&& other) noexcept
{
    if (this != &other)
    {
        if (m_programId != 0)
        {
            glDeleteProgram(m_programId);
        }
        
        m_programId = other.m_programId;
        m_name = std::move(other.m_name);
        m_uniformCache = std::move(other.m_uniformCache);
        
        other.m_programId = 0;
    }
    return *this;
}

void ShaderBase::use()
{
    glUseProgram(m_programId);
}

void ShaderBase::setBool(const std::string& name, bool value) const
{
    glUniform1i(getUniformLocation(name), static_cast<int>(value));
}

void ShaderBase::setInt(const std::string& name, int value) const
{
    glUniform1i(getUniformLocation(name), value);
}

void ShaderBase::setUint(const std::string& name, unsigned int value) const
{
    glUniform1ui(getUniformLocation(name), value);
}

void ShaderBase::setFloat(const std::string& name, float value) const
{
    glUniform1f(getUniformLocation(name), value);
}

void ShaderBase::setVec2(const std::string& name, const glm::vec2& value) const
{
    glUniform2fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setVec2(const std::string& name, float x, float y) const
{
    glUniform2f(getUniformLocation(name), x, y);
}

void ShaderBase::setVec3(const std::string& name, const glm::vec3& value) const
{
    glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setVec3(const std::string& name, float x, float y, float z) const
{
    glUniform3f(getUniformLocation(name), x, y, z);
}

void ShaderBase::setVec4(const std::string& name, const glm::vec4& value) const
{
    glUniform4fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setVec4(const std::string& name, float x, float y, float z, float w) const
{
    glUniform4f(getUniformLocation(name), x, y, z, w);
}

void ShaderBase::setVec2I(const std::string& name, const glm::ivec2& value) const
{
    glUniform2iv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setVec2I(const std::string& name, int x, int y) const
{
    glUniform2i(getUniformLocation(name), x, y);
}

void ShaderBase::setVec3I(const std::string& name, const glm::ivec3& value) const
{
    glUniform3iv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setVec3I(const std::string& name, int x, int y, int z) const
{
    glUniform3i(getUniformLocation(name), x, y, z);
}

void ShaderBase::setVec4I(const std::string& name, const glm::ivec4& value) const
{
    glUniform4iv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void ShaderBase::setMat2(const std::string& name, const glm::mat2& value) const
{
    glUniformMatrix2fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void ShaderBase::setMat3(const std::string& name, const glm::mat3& value) const
{
    glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void ShaderBase::setMat4(const std::string& name, const glm::mat4& value) const
{
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

std::string ShaderBase::readShaderFile(const std::string& filePath) const
{
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try
    {
        file.open(filePath);
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        return stream.str();
    }
    catch (const std::ifstream::failure& e)
    {
        throw std::runtime_error("Failed to read shader file '" + filePath + "': " + e.what());
    }
}

GLuint ShaderBase::compileShader(const std::string& source, GLenum shaderType, 
                                 const std::string& shaderName) const
{
    GLuint shader = glCreateShader(shaderType);
    const char* sourcePtr = source.c_str();
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);
    
    checkErrors(shader, "SHADER", shaderName);
    
    return shader;
}

void ShaderBase::createProgram()
{
    m_programId = glCreateProgram();
}

void ShaderBase::attachShader(GLuint shaderId)
{
    glAttachShader(m_programId, shaderId);
}

void ShaderBase::linkProgram()
{
    glLinkProgram(m_programId);
    checkErrors(m_programId, "PROGRAM", m_name);
}

void ShaderBase::cleanupShader(GLuint shaderId)
{
    glDetachShader(m_programId, shaderId);
    glDeleteShader(shaderId);
}

GLint ShaderBase::getUniformLocation(const std::string& name) const
{
    auto it = m_uniformCache.find(name);
    if (it != m_uniformCache.end())
    {
        return it->second;
    }
    
    GLint location = glGetUniformLocation(m_programId, name.c_str());
    m_uniformCache[name] = location;
    
    if (location == -1)
    {
        // Warning: uniform not found (could be optimized out by compiler)
        // std::cerr << "Warning: Uniform '" << name << "' not found in shader '" << m_name << "'" << std::endl;
    }
    
    return location;
}

void ShaderBase::checkErrors(GLuint object, const std::string& type, 
                             const std::string& name) const
{
    GLint success;
    char infoLog[1024];
    
    if (type == "SHADER")
    {
        glGetShaderiv(object, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(object, 1024, nullptr, infoLog);
            throw std::runtime_error("Shader compilation error in '" + name + "':\n" + infoLog);
        }
    }
    else // PROGRAM
    {
        glGetProgramiv(object, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(object, 1024, nullptr, infoLog);
            throw std::runtime_error("Program linking error in '" + name + "':\n" + infoLog);
        }
    }
}
