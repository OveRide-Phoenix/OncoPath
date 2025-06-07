from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Ensure the directory exists
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

@app.route('/get_image/<filename>')
def get_image(filename):
    """
    Serve images from the temp_images folder.
    """
    return send_from_directory(TEMP_IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
