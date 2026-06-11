import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("outputs/dogs_vs_cats_cnn.keras")

# Class names (adjust if your training order is different)
class_names = ["Cat", "Dog"]

# Input image
img_path = input("Enter image path: ")

if not os.path.exists(img_path):
    print("Image not found!")
    exit()

# Preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array, verbose=0)

# Get probability
prob = float(prediction[0][0])

# Binary classification
if prob > 0.5:
    predicted_class = "Dog"
    confidence = prob * 100
else:
    predicted_class = "Cat"
    confidence = (1 - prob) * 100

# Display results
print("\n" + "=" * 40)
print("DOG vs CAT CLASSIFICATION RESULT")
print("=" * 40)
print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")
print(f"Dog Probability : {prob * 100:.2f}%")
print(f"Cat Probability : {(1 - prob) * 100:.2f}%")
print("=" * 40)
