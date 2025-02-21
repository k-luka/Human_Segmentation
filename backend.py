import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch

app = Flask(__name__)

# Load the TorchScript DeepLabV3 model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('deeplabv3_person_parts_traced.pt', map_location=device)
model.eval()

# Define a fixed color palette for 5 classes:
# 0: background, 1: head, 2: torso, 3: arms, 4: legs
PALETTE_5 = np.array([
    (0, 0, 0),       # class 0 - background: black
    (128, 0, 0),     # class 1 - head: maroon
    (0, 128, 0),     # class 2 - torso: green
    (0, 0, 128),     # class 3 - arms: navy
    (128, 0, 128)    # class 4 - legs: purple
], dtype=np.uint8)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    """Handle image upload, resize for model input, perform segmentation, and return results."""
    # Ensure a file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image and convert to RGB; keep a copy of the original image for overlay
    original_image = Image.open(file.stream).convert('RGB')
    orig_width, orig_height = original_image.size

    # Resize the image for inference if needed: maximum dimension 800
    max_dim = 512
    if orig_width > max_dim or orig_height > max_dim:
        scale_factor = min(max_dim / orig_width, max_dim / orig_height)
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        resized_image = original_image.resize((new_width, new_height), resample=Image.LANCZOS)
    else:
        resized_image = original_image.copy()
    
    # Preprocess the resized image for the model
    img_array = np.array(resized_image).astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std
    img_array[:, :, 0] = (img_array[:, :, 0] - 0.485) / 0.229
    img_array[:, :, 1] = (img_array[:, :, 1] - 0.456) / 0.224
    img_array[:, :, 2] = (img_array[:, :, 2] - 0.406) / 0.225
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Run the model on the resized image to get the segmentation output
    with torch.no_grad():
        output = model(img_tensor)
    if isinstance(output, dict):
        out_tensor = output.get('out', list(output.values())[0])
    elif isinstance(output, (list, tuple)):
        out_tensor = output[0]
    else:
        out_tensor = output
    # Predicted mask (on resized image) with values 0-4
    mask_tensor = out_tensor.squeeze(0).argmax(dim=0).to('cpu')
    mask_small = mask_tensor.numpy().astype(np.int32)

    # Resize the mask back to the original image resolution using nearest-neighbor
    mask_pil_small = Image.fromarray(mask_small.astype(np.uint8))
    mask_pil = mask_pil_small.resize((orig_width, orig_height), resample=Image.NEAREST)
    mask = np.array(mask_pil).astype(np.int32)

    # Create the segmentation mask image using the fixed 5-class palette
    color_mask = PALETTE_5[mask]  # shape (H, W, 3)
    mask_img = Image.fromarray(color_mask.astype(np.uint8))

    # Generate overlay image: blend the original image with the segmentation mask overlay
    original_rgba = original_image.convert('RGBA')
    overlay_rgba = Image.fromarray(color_mask.astype(np.uint8)).convert('RGBA')
    # Set alpha channel: 128 (~50% transparency) for non-background pixels, 0 for background
    alpha_mask = (mask > 0).astype(np.uint8) * 128
    alpha_channel = Image.fromarray(alpha_mask, mode='L')
    overlay_rgba.putalpha(alpha_channel)
    overlay_result = Image.alpha_composite(original_rgba, overlay_rgba).convert('RGB')

    # Encode images to base64 strings for display in the frontend
    buf1 = BytesIO()
    overlay_result.save(buf1, format='PNG')
    overlay_data = base64.b64encode(buf1.getvalue()).decode('utf-8')
    buf2 = BytesIO()
    mask_img.save(buf2, format='PNG')
    mask_data = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return jsonify({
        'overlay': 'data:image/png;base64,' + overlay_data,
        'mask': 'data:image/png;base64,' + mask_data
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
