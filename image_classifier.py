# Image Classification Using a Pretrained Model
# CodeOrbit Tech - Artificial Intelligence Internship

import torch
from torchvision import models
from PIL import Image

# Load the pretrained ResNet18 model
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Set the model to evaluation mode
model.eval()

# Get the preprocessing steps
preprocess = weights.transforms()

# Get the class names
categories = weights.meta["categories"]


# Function to classify an image
def classify_image(image_name):

    # Open the image
    image = Image.open(image_name).convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get predicted class
    predicted_class = output.argmax(1).item()

    # Get predicted label
    predicted_label = categories[predicted_class]

    print("Image:", image_name)
    print("Predicted Label:", predicted_label)
    print("--------------------------------")


# Display project title
print("========================================")
print("       IMAGE CLASSIFICATION")
print("========================================")
print("Model: Pretrained ResNet18")
print()

# Classify the first image
classify_image("sample.jpeg")

# Classify the second image
classify_image("cat.jpeg")