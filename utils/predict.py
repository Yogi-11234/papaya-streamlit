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
        )[0]

        confidence, prediction = torch.max(
            probabilities,
            dim=0
        )

    all_probs = {}

    for cls, prob in zip(CLASS_NAMES, probabilities):
        all_probs[cls] = round(
            float(prob.item() * 100),
            2
        )

    return {
        "class": CLASS_NAMES[prediction.item()],
        "confidence": float(confidence.item() * 100),
        "all_probs": all_probs
    }