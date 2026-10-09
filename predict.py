import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("handwritten_digit_model.keras")

# Load your image
image = Image.open("my_digit.png").convert("L")

# Resize to MNIST size
image = image.resize((28, 28))

# Convert to array
image = np.array(image)

# MNIST uses white digit on black background
image = 255 - image

# Normalize
image = image.astype("float32") / 255.0

# Add dimensions
image = image.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(image, verbose=0)

digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("Predicted digit:", digit)
print(f"Confidence: {confidence:.2f}%")