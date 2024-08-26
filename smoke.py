from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from keras.layers import DepthwiseConv2D
app = Flask(__name__)
def custom_depthwise_conv2d(**kwargs):
    return DepthwiseConv2D(**kwargs)

# Load the model
model = load_model("keras_model.h5", compile=False, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})

# Load class names
with open("labels.txt", "r") as f:
    class_names = f.readlines()

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

        return jsonify({
            'class': class_name,
            'confidence_score': float(confidence_score)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
