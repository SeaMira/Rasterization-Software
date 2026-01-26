#ifndef _PIPELINE_CONFIG_H_
#define _PIPELINE_CONFIG_H_

#include "ui/component.h"
#include "ui/element.h"

/**
 * @class PipelineConfigComponent
 * @brief A UI component for configuring hybrid pipeline parameters.
 * 
 * This component provides sliders and controls for adjusting:
 * - Small entity threshold (pixels^2)
 * - Max entities per tile
 * - Tile size (display only)
 */
class PipelineConfigComponent : public Component
{
public:
    /**
     * @brief Constructs a PipelineConfigComponent with the specified parameters.
     * 
     * @param name The name of the component.
     * @param smallThreshold Reference to the small entity threshold value.
     * @param tileSize Reference to the tile size value (display only).
     * @param maxEntitiesPerTile Reference to max entities per tile (adjustable).
     * @param tilesX Reference to number of tiles in X (display only).
     * @param tilesY Reference to number of tiles in Y (display only).
     */
    PipelineConfigComponent(std::string name, 
                            int& smallThreshold,
                            int& tileSize,
                            int& maxEntitiesPerTile,
                            int& tilesX,
                            int& tilesY)
        : Component(name)
    {
        // Section: Classification
        addElement(std::make_unique<TextElement>("--- Classification ---"));
        addElement(std::make_unique<SliderElementI>("Small Threshold (px^2)", smallThreshold, 16, 4096));
        
        // Section: Tile Configuration
        addElement(std::make_unique<NewLineElement>());
        addElement(std::make_unique<TextElement>("--- Tile Config ---"));
        addElement(std::make_unique<TextElementi>("Tile Size", tileSize));
        addElement(std::make_unique<SliderElementI>("Max Entities/Tile", maxEntitiesPerTile, 64, 512));
        addElement(std::make_unique<TextElementi>("Tiles X", tilesX));
        addElement(std::make_unique<TextElementi>("Tiles Y", tilesY));
    }
};

#endif // _PIPELINE_CONFIG_H_
