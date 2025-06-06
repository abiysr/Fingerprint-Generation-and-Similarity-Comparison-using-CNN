import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
import cv2

# Load trained CNN model
my_cnn_model = load_model('my_cnn_model.h5')

# Function to generate fingerprint for an image with resizing
def generate_fingerprint_from_image(image_path, model, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    fingerprint = model.predict(img_array)
    return fingerprint.flatten()

# Function to generate fingerprint for an image without resizing
def generate_fingerprint_from_array(image_array, model):
    img_array = np.expand_dims(image_array, axis=0)
    fingerprint = model.predict(img_array)
    return fingerprint.flatten()

# Example usage with resized images
image1_path = '1A.jpeg'
image2_path = '2B.jpeg'

image1_resized = cv2.resize(cv2.imread(image1_path), (150, 150))
image2_resized = cv2.resize(cv2.imread(image2_path), (150, 150))

fingerprint1_resized = generate_fingerprint_from_array(image1_resized, my_cnn_model)
fingerprint2_resized = generate_fingerprint_from_array(image2_resized, my_cnn_model)

# Calculate Euclidean distance between the resized fingerprints
distance = euclidean_distances([fingerprint1_resized], [fingerprint2_resized])
# Convert distance to similarity score
similarity_score_resized = 1 / (1 + distance)  
print("Euclidean Distance (Resized):", distance[0][0])
print("Similarity Score (Resized):", similarity_score_resized[0][0])
if similarity_score_resized[0][0] > 0.8:
    print("High Similarity")
else:
    print("Low Similarity")
