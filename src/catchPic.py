from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data['image']
    # Strip the header from the base64 encoded string
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image.save("captured_image.png")  # Save the image to a file
    

    return jsonify({'message': 'Image uploaded successfully!'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5000)  # Run on all network interfaces
