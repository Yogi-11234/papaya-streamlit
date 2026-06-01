import torch

from utils.image_processing import preprocess_image

CLASS_NAMES = [
    "Anthracnose",
    "BacterialSpot",
    "Curl",
    "Healthy",
    "RingSpot"
]


def predict_image(model, image_path):

    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "class": CLASS_NAMES[prediction.item()],
        "confidence": float(confidence.item() * 100)
    }