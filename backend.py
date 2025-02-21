import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch

app = Flask(__name__)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load both models at startup
# ResNet-based model
model_resnet = torch.jit.load('deeplabv3_person_parts_traced.pt', map_location=device)
model_resnet.eval()

# MobileNet-based model
model_mobilenet = torch.jit.load('deeplabv3_mobilenet_person_parts_traced.pt', map_location=device)
model_mobilenet.eval()

# Fixed color palette for 5 classes: 0: bg, 1: head, 2: torso, 3: arms, 4: legs
PALETTE_5 = np.array([
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 0, 128)
], dtype=np.uint8)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    # Determine model type from JSON, form data, or query string
    if request.is_json:
        data = request.get_json()
        print("Received JSON payload:", data)
        model_type = data.get('model', 'mobilenet').lower()
    elif 'file' in request.files and request.files['file'].filename != '':
        # Read model from form data for file uploads
        model_type = request.form.get('model', 'mobilenet').lower()
    else:
        model_type = request.args.get('model', 'resnet').lower()
    
    print("Selected model:", model_type)

    # Select model based on model_type
    if model_type == 'resnet':
        current_model = model_resnet
    else:
        current_model = model_mobilenet

    # Handle image input (file upload or JSON base64)
    if request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided in JSON'}), 400
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        try:
            image_bytes = base64.b64decode(img_data)
            original_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    elif 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        original_image = Image.open(file.stream).convert('RGB')
    else:
        return jsonify({'error': 'No valid image provided'}), 400

    orig_width, orig_height = original_image.size

    # Resize for inference (max dimension 800)
    max_dim = 800
    if orig_width > max_dim or orig_height > max_dim:
        scale_factor = min(max_dim / orig_width, max_dim / orig_height)
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        resized_image = original_image.resize((new_width, new_height), resample=Image.LANCZOS)
    else:
        resized_image = original_image.copy()

    # Preprocess: convert to normalized tensor
    img_array = np.array(resized_image).astype(np.float32) / 255.0
    img_array[:, :, 0] = (img_array[:, :, 0] - 0.485) / 0.229
    img_array[:, :, 1] = (img_array[:, :, 1] - 0.456) / 0.224
    img_array[:, :, 2] = (img_array[:, :, 2] - 0.406) / 0.225
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = current_model(img_tensor)
    if isinstance(output, dict):
        out_tensor = output.get('out', list(output.values())[0])
    elif isinstance(output, (list, tuple)):
        out_tensor = output[0]
    else:
        out_tensor = output

    mask_tensor = out_tensor.squeeze(0).argmax(dim=0).to('cpu')
    mask_small = mask_tensor.numpy().astype(np.int32)

    # Resize mask back to original resolution
    mask_pil_small = Image.fromarray(mask_small.astype(np.uint8))
    mask_pil = mask_pil_small.resize((orig_width, orig_height), resample=Image.NEAREST)
    mask = np.array(mask_pil).astype(np.int32)

    color_mask = PALETTE_5[mask]
    mask_img = Image.fromarray(color_mask.astype(np.uint8))

    # Generate overlay image
    original_rgba = original_image.convert('RGBA')
    overlay_rgba = Image.fromarray(color_mask.astype(np.uint8)).convert('RGBA')
    alpha_mask = (mask > 0).astype(np.uint8) * 128
    alpha_channel = Image.fromarray(alpha_mask, mode='L')
    overlay_rgba.putalpha(alpha_channel)
    overlay_result = Image.alpha_composite(original_rgba, overlay_rgba).convert('RGB')

    # Encode to base64
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
