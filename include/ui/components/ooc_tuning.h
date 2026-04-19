#ifndef _OOC_TUNING_H_
#define _OOC_TUNING_H_

#include "ui/component.h"
#include "ui/element.h"

/** Slider with fine-grained format; logarithmic scale suits thresholds spanning orders of magnitude. */
class OocVisibilityThresholdSliderElement : public Element
{
public:
    OocVisibilityThresholdSliderElement(std::string label, float& value)
        : m_label(std::move(label)), m_value(&value) {}

    void render() override
    {
        ImGui::SliderFloat(m_label.c_str(), m_value, 1e-10f, 1.0f, "%.5f",
                           ImGuiSliderFlags_Logarithmic);
    }

private:
    std::string m_label;
    float*      m_value;
};

/**
 * Runtime tuning for the out-of-core CUDA pipeline (values read each frame into OocConstants).
 */
class OocTuningComponent : public Component
{
public:
    OocTuningComponent(std::string name, float& visibilityThreshold)
        : Component(std::move(name))
    {
        addElement(std::make_unique<TextElement>(
            "visibility_threshold: probabilistic / HiZ tile tests (see outofcore_kernels)"));
        addElement(std::make_unique<OocVisibilityThresholdSliderElement>(
            "Visibility threshold", visibilityThreshold));
    }
};

#endif
