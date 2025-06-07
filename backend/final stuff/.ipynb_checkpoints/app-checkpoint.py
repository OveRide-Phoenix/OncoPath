from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import torch
from your_model_file import UNet, analyze_tumor_properties  # Import your model and analysis function

app = Flask(__name__)

# Load the UNet model
model_path = r"path_to_your_model.pth"  # Replace with the path to your .pth file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Load and preprocess the image
    image = Image.open(file).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match the model input
    image_tensor = torch.tensor(np.array(image) / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)

    # Predict the tumor mask
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask_np = pred_mask.cpu().squeeze().numpy()

    # Analyze tumor properties
    image_np = np.array(image)
    properties = analyze_tumor_properties(pred_mask_np, image_np)

    return jsonify(properties)

if __name__ == '__main__':
    app.run(debug=True)
