import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import base64
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend before importing pyplot


# Define the UNet model architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        out = torch.sigmoid(out)
        return out


# Flask app
app = Flask(__name__, static_folder='.')

model_path = r"best_model_epoch88_dice0.7369.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


@app.route('/')
def index():
    return send_from_directory('.', 'indexfinal2.html')


# BI-RADS Analysis Function
def analyze_tumor_properties(mask, image):
    binary_mask = (mask > 0.5).astype(np.uint8)
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)

    if len(regions) == 0:
        return {
            "error": "No tumor detected",
            "tumor_details": []
        }

    tumor_details = []
    pixel_to_mm2 = 0.1  # Adjust based on actual scaling factor

    for region in regions:
        tumor_area = region.area
        tumor_size_mm2 = tumor_area * pixel_to_mm2

        eccentricity = region.eccentricity
        solidity = region.solidity
        shape = "Round" if eccentricity < 0.4 else "Oval" if eccentricity < 0.8 else "Irregular"
        margin = "Circumscribed" if solidity > 0.9 else "Indistinct" if solidity > 0.7 else "Spiculated"

        coords = region.coords
        tumor_intensity = np.mean([image[y, x] for y, x in coords])

        birads_category = 1
        if shape == "Irregular" or margin in ["Indistinct", "Spiculated"]:
            birads_category = 4
        elif margin == "Indistinct":
            birads_category = 3
        birads_analysis = f"Category - {birads_category}"

        cancer_stage = "Stage 0"
        if tumor_size_mm2 > 50:
            cancer_stage = "Stage II"
        elif tumor_size_mm2 > 20:
            cancer_stage = "Stage I"

        diagnostic_summary = f"Tumor with {shape} shape and {
            margin} margins. Predicted stage: {cancer_stage}."

        tumor_details.append({
            "Size (mm²)": tumor_size_mm2,
            "Size (pixels)": tumor_area,
            "Shape": shape,
            "Margin": margin,
            "Opacity": tumor_intensity,
            "BI-RADS": birads_analysis,
            "Cancer Stage": cancer_stage,
            "Diagnostic Summary": diagnostic_summary
        })

    return tumor_details


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    cmap_option = request.form.get("colormap", "inferno")

    # Ensure the colormap is valid
    valid_cmaps = ["gray", "inferno", "nipy_spectral"]
    if cmap_option not in valid_cmaps:
        cmap_option = "inferno"  # Default to inferno if invalid input

    image = Image.open(file).convert('L')
    image = image.resize((128, 128))
    image_tensor = torch.tensor(
        np.array(image) / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)

    intermediate_images = {}

    with torch.no_grad():
        e1 = model.enc1(image_tensor)
        e2 = model.enc2(F.max_pool2d(e1, 2))
        e3 = model.enc3(F.max_pool2d(e2, 2))
        e4 = model.enc4(F.max_pool2d(e3, 2))
        bottleneck = model.bottleneck(F.max_pool2d(e4, 2))

        d4 = model.upconv4(bottleneck)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = model.dec4(d4)

        d3 = model.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = model.dec3(d3)

        d2 = model.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = model.dec2(d2)

        d1 = model.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = model.dec1(d1)

        pred_mask = model.out_conv(d1).cpu().squeeze().numpy()

        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        fig, ax = plt.subplots(facecolor='#112a40')
        ax.set_facecolor('#112a40')
        ax.imshow(image, cmap=cmap_option)
        ax.contour(binary_mask, colors='w')
        ax.axis("on")
        # Change tick labels to white
        ax.tick_params(axis='both', colors='white')
        ax.spines['top'].set_color('white')  # Top border
        ax.spines['bottom'].set_color('white')  # Bottom border
        ax.spines['left'].set_color('white')  # Left border
        ax.spines['right'].set_color('white')  # Right border
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        segmentation_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        for stage, img_data in zip(["encoder", "bottleneck", "decoder"], [e1, bottleneck, d1]):
            fig, ax = plt.subplots()
            ax.imshow(img_data[0, 0], cmap="gray")  # Select the first channel
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            intermediate_images[stage] = base64.b64encode(
                buf.getvalue()).decode("utf-8")
            plt.close(fig)

        tumor_details = analyze_tumor_properties(pred_mask, np.array(image))

    return jsonify({
        "segmentation_image": segmentation_image,
        "intermediate_images": intermediate_images,
        "tumor_details": tumor_details,
    })


if __name__ == '__main__':
    app.run(debug=False)
