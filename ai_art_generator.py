import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import argparse

def load_and_prepare_image(image_path, size=(512, 512)):
    """Load an image file and prepare it for the model."""
    img = load_img(image_path, target_size=size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

def generate_art(model, image):
    """Generate art using the pre-trained model and an input image."""
    prediction = model.predict(image)
    return (prediction[0] * 255).astype('uint8')

def save_image(image, path):
    """Save the generated image to disk."""
    img = Image.fromarray(image)
    img.save(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate AI art from your photo.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    model = load_model('model.h5')  # Path to the pre-trained model

    image = load_and_prepare_image(args.image_path)
    art_image = generate_art(model, image)
    save_image(art_image, 'output.jpg')
    print("Art generated successfully! Check the output.jpg file.")
