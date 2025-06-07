from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.measure import label, regionprops
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define UNet model
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
        return torch.sigmoid(out)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("best_model_epoch88_dice0.7369.pth", map_location=device))
model.to(device)
model.eval()

# Tumor analysis
def analyze_tumor_properties(mask, image):
    binary_mask = (mask > 0.5).astype(np.uint8)
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    if len(regions) == 0:
        return []

    region = regions[0]
    pixel_to_mm2 = 0.1
    area = region.area
    eccentricity = region.eccentricity
    solidity = region.solidity

    shape = "Round" if eccentricity < 0.4 else "Oval" if eccentricity < 0.8 else "Irregular"
    margin = "Circumscribed" if solidity > 0.9 else "Indistinct" if solidity > 0.7 else "Spiculated"
    tumor_size_mm2 = area * pixel_to_mm2

    stage = "Stage 0"
    if tumor_size_mm2 > 50:
        stage = "Stage II"
    elif tumor_size_mm2 > 20:
        stage = "Stage I"

    return [{
        "Size (mm²)": tumor_size_mm2,
        "Size (pixels)": area,
        "Shape": shape,
        "Margin": margin,
        "BI-RADS": "BI-RADS 4" if shape == "Irregular" or margin != "Circumscribed" else "BI-RADS 2",
        "Cancer Stage": stage,
    }]

def suggest_treatments(stage: str, birads: str):
    treatments = []

    # Base treatment for all stages
    if stage in ["Stage 0", "Stage I", "Stage II"]:
        treatments.append({
            "type": "Surgery",
            "recommendation": "Lumpectomy with sentinel lymph node biopsy",
            "priority": "Primary",
            "description": "Breast-conserving surgery to remove the tumor while preserving breast tissue"
        })

    if stage == "Stage II":
        treatments.append({
            "type": "Chemotherapy",
            "recommendation": "Adjuvant chemotherapy (AC-T protocol)",
            "priority": "Secondary",
            "description": "Systemic treatment to eliminate any remaining cancer cells"
        })

    if birads == "BI-RADS 4" or stage in ["Stage I", "Stage II"]:
        treatments.append({
            "type": "Radiation",
            "recommendation": "Whole breast radiation therapy",
            "priority": "Secondary",
            "description": "Targeted radiation to reduce local recurrence risk"
        })

    # Optional: Add hormone therapy for completeness
    if birads == "BI-RADS 4" and stage == "Stage II":
        treatments.append({
            "type": "Hormone Therapy",
            "recommendation": "Tamoxifen for estrogen receptor-positive tumors",
            "priority": "Optional",
            "description": "Helps prevent cancer recurrence by blocking hormones"
        })

    return treatments


# Upload route
@app.post("/upload")
async def upload(file: UploadFile = File(...), colormap: str = Form("inferno")):
    image = Image.open(BytesIO(await file.read())).convert("L")
    image = image.resize((128, 128))
    image_tensor = torch.tensor(np.array(image) / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        # Get intermediate layer outputs manually
    with torch.no_grad():
        e1 = model.enc1(image_tensor)
        e2 = model.enc2(F.max_pool2d(e1, 2))
        e3 = model.enc3(F.max_pool2d(e2, 2))
        e4 = model.enc4(F.max_pool2d(e3, 2))
        b = model.bottleneck(F.max_pool2d(e4, 2))
        d4 = model.upconv4(b)
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
        output = model.out_conv(d1)
        mask = torch.sigmoid(output).cpu().squeeze().numpy()
        mask = output.cpu().squeeze().numpy()

    # Generate segmentation overlay image (base64)
    # Generate segmentation overlays for multiple colormaps
    binary_mask = (mask > 0.5).astype(np.uint8)
    colormaps = ["inferno", "nipy_spectral", "gray"]

    segmentation_images = {}
    for cmap in colormaps:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap)
        ax.contour(binary_mask, colors="w")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        seg_image_b64 = base64.b64encode(buf.read()).decode("utf-8")
        segmentation_images[cmap] = seg_image_b64

    tumor_details = analyze_tumor_properties(mask, np.array(image))
    tumor_stage = tumor_details[0].get("Cancer Stage", "Unknown")
    birads = tumor_details[0].get("BI-RADS", "BI-RADS N/A")
    treatments = suggest_treatments(tumor_stage, birads)

    intermediate_images = {}
    for stage, img_data in zip(["encoder", "bottleneck", "decoder"], [e1, b, d1]):
        fig, ax = plt.subplots()
        ax.imshow(img_data[0, 0].cpu(), cmap="gray")  # only first channel
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        intermediate_images[stage] = base64.b64encode(buf.read()).decode("utf-8")

    return JSONResponse(content={
    "segmentation": {
        "segmentedImages": segmentation_images,
    },
    "tumor_details": tumor_details,
    "pipeline": {
    "encoder": {
        "name": "Encoder",
        "description": "Feature extraction from input image",
        "image": f"data:image/png;base64,{intermediate_images.get('encoder')}"
    },
    "bottleneck": {
        "name": "Bottleneck",
        "description": "Compressed representation of features",
        "image": f"data:image/png;base64,{intermediate_images.get('bottleneck')}"
    },
    "decoder": {
        "name": "Decoder",
        "description": "Reconstruction of segmentation mask",
        "image": f"data:image/png;base64,{intermediate_images.get('decoder')}"
    }
},
    "treatments": treatments,
})
