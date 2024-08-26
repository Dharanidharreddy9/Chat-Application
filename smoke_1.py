from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
from keras.layers import DepthwiseConv2D
import os

app = Flask(__name__)

def custom_depthwise_conv2d(**kwargs):
    return DepthwiseConv2D(**kwargs)

# Load the model
model = load_model("keras_model.h5", compile=False, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})

# Load class names
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Directory to save images with captions
SAVE_DIR = r'D:\smoke_detection\smoker_detection\captioned_images'
os.makedirs(SAVE_DIR, exist_ok=True)


def add_caption(image, caption):
    """ Add a caption to the image. """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    width, height = image.size
    # Position text at the bottom-left corner
    text_x = 10
    text_y = height - text_height - 10
    draw.text((text_x, text_y), caption, font=font, fill="white")
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file
        image = Image.open(file.stream).convert("RGB")
        
        # Prepare the image data
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # If smoke is detected, add a caption
        if 'smoking' in class_name.lower():  # Modify this condition based on your class names
            caption = "Smoking is injurious to health"

            image = add_caption(image, caption)
            # Save the image with caption

            file_path = os.path.join(SAVE_DIR, file.filename)
            image.save(file_path)
            return jsonify({
                'class': class_name,
                'confidence_score': float(confidence_score),
                'image_path': file_path
            })

        return jsonify({
            'class': class_name,
            'confidence_score': float(confidence_score),
            'image_path': None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
