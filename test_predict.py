from utils.model_loader import load_model
from utils.predict import predict_image

model = load_model(
    "model/papayaleaf_cnn_5class.pth"
)

result = predict_image(
    model,
    "assets/example.jpg"
)

print(result)