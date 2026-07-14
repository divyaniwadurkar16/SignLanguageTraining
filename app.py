import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('sign_language_model.keras')

# Define the class labels (assuming alphabetical order from 'A' to 'Z' excluding 'J', 'Z', 'space', 'nothing')
# Based on the number of classes (28) and the common ASL alphabet datasets
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
    'del', 'nothing', 'space'
]

# Adjusting class_names to match the 28 labels if 'del' is included or 'J' is excluded.
# For this dataset, let's assume the 28 classes are derived from the 28 letters/gestures found.
# If your class mapping is different, you'll need to update `class_names` accordingly.
# The ImageDataGenerator assigns integer labels based on subdirectory names in alphabetical order.

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array /= 255.0 # Rescale the image
    return img_array

# Function to predict the sign
def predict_sign(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    # Make sure class_names is correctly defined to match the model's output order
    return class_names[predicted_class_index], predictions[0][predicted_class_index]

if __name__ == '__main__':
    print("Model loaded. Use predict_sign(img_path) to make predictions.")
    # Example usage (you would replace 'path/to/your/image.jpg' with an actual image path):
    # try:
    #     # Assuming you have an image file, e.g., 'A_test.jpg' in your /content directory
    #     # You can upload one or use an existing one from the extracted dataset
    #     sample_image_path = '/content/asl_alphabet_test_extracted/asl_alphabet_test/A/A_test.jpg'
    #     predicted_sign, confidence = predict_sign(sample_image_path)
    #     print(f"Predicted sign: {predicted_sign} with confidence: {confidence:.2f}")
    # except FileNotFoundError:
    #     print("Sample image not found. Please provide a valid path to an image.")
