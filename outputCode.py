import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from typing import Tuple
import numpy as np
from vision_agent.tools import (
    countgd_sam2_object_detection,
    overlay_bounding_boxes,
    save_image
)

def detect_empty_parking_spaces(
    image: np.ndarray,
    output_image_path: str = 'marked_empty_spaces.png',
) -> Tuple[np.ndarray, list]:
    """
    Detects and marks empty parking spaces in an image, saving a marked image
    and return the coordinates.

    Parameters:
        image (np.ndarray): Input image to process
        output_image_path (str): Path to save the marked image (default: 'marked_empty_spaces.png')

    Returns:
        Tuple[np.ndarray, list]: Returns a tuple containing:
            - The marked image with empty spaces highlighted
            - Empty space coordinates
    """
    # 1. Detect cars
    detections = countgd_sam2_object_detection("car", image)

    # 2. Process detections to find empty spaces
    sorted_detections = sorted(detections, key=lambda x: (x["bbox"][1], x["bbox"][0]))

    horizontal_lines = []
    while len(sorted_detections) > 0:
        current = sorted_detections[0]
        x_min, y_min, x_max, y_max = current["bbox"]
        mean_y = (y_min + y_max) / 2
        line = [det for det in sorted_detections if abs((det["bbox"][1] + det["bbox"][3])/2 - mean_y) < 0.1]
        horizontal_lines.append(line)
        
        for det in line:
            sorted_detections.remove(det)

    gaps = []
    for line in horizontal_lines:
        line = sorted(line, key=lambda x: x["bbox"][0])
        median_width = np.median([line[i]["bbox"][2] - line[i]["bbox"][0] for i in range(len(line))])
        median_height = np.median([line[i]["bbox"][3] - line[i]["bbox"][1] for i in range(len(line))])
        
        for i in range(len(line) - 1):
            w_gap = line[i + 1]["bbox"][0] - line[i]["bbox"][2]
            if w_gap > (0.5 * median_width):
                count = np.round(w_gap / median_width)
                for j in range(int(count)):
                    gap = [
                        float(line[i]["bbox"][2] + j * median_width),
                        float(line[i]["bbox"][1]),
                        float(line[i]["bbox"][2] + (j + 1) * median_width),
                        float(line[i]["bbox"][1] + median_height)
                    ]
                    gaps.append(gap)

    # 3. Create bounding boxes for visualization
    empty_spaces = [{"label": "empty_space", "score": 1.0, "bbox": gap} for gap in gaps]
    image_with_boxes = overlay_bounding_boxes(image, empty_spaces)

    # 4. format the coordinates of empty spaces
    empty_space_coordinates = {
        'empty_spaces': [
            {
                'coordinates': {
                    'x_min': gap[0],
                    'y_min': gap[1],
                    'x_max': gap[2],
                    'y_max': gap[3]
                }
            }
            for gap in gaps
        ]
    }

    return image_with_boxes, empty_space_coordinates