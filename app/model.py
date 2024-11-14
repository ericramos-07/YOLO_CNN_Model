from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from sklearn.metrics import average_precision_score
import numpy as np

# Function to load the YOLO model
def load_model(model_path: str):
    """
    Load the YOLO model from the provided .pt file.
    """
    model = YOLO(model_path)
    return model

def predict(model, file):
    """
    Given a model and an image file, this function will return the classification of the mango,
    texture, blemishes.
    """
    # Use BytesIO to handle the in-memory image file
    image = Image.open(BytesIO(file.read()))

    # Run inference with the model
    results = model(image)  # This returns a list of results objects

    # Extract the first result (assuming only one image is processed at a time)
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        mango_type = None
        confidence = None
        texture = None
        blemishes = False
        all_class_names = []
        all_confidences = []

        # Iterate over detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names.get(class_id, "Unknown Class").upper()
            box_confidence = float(box.conf.item())

            # Collect all class names and their confidences for later average precision calculation
            all_class_names.append(class_name)
            all_confidences.append(box_confidence)

            # Find mango type with the highest confidence
            if "CLASS" in class_name:
                if not mango_type or box_confidence > confidence:
                    mango_type = class_name
                    confidence = box_confidence

            # Detect texture based on explicit "Smooth-Texture" or "Rough-Texture" in class name
            if "SMOOTH-TEXTURE" in class_name:
                texture = "Smooth"
            elif "ROUGH-TEXTURE" in class_name:
                texture = "Rough"

            # Detect blemish presence
            if "BLEMISH" in class_name:
                blemishes = True

        # If no mango type detected, return error
        if not mango_type:
            return {"error": "No mango detected"}

        # Default texture to "Unknown" if not identified
        texture = texture or "Unknown"

        """
        # Calculate average precision if there are any valid detections
        if all_class_names and all_confidences:
            # Calculate average precision for mango detection
            # Note: You need true labels (ground truth) and predicted labels for AP calculation
            # For now, we'll use all detections' confidences as predictions for AP score
            ap_score = average_precision_score(np.ones_like(all_confidences), all_confidences)  # Simplified AP for illustration
        else:
            ap_score = 0
        """

        # Return the predicted results in the desired format
        return {
            "confidence": confidence,
            #"average_precision": ap_score,
            "mango_type": mango_type,
            "blemishes": blemishes,
            "texture": texture
        }

    else:
        return {"error": "No objects detected"}
