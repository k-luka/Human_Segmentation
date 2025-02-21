// --- File Upload Code (updated) ---
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview');
const overlayImg = document.getElementById('overlay-img');
const maskImg = document.getElementById('mask-img');
const resultContainer = document.getElementById('result-container');
const uploadBtn = document.getElementById('upload-btn');
const modelSelect = document.getElementById('model-select');  // use same dropdown for both

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    previewImg.src = URL.createObjectURL(file);
    previewImg.style.display = 'block';
    resultContainer.style.display = 'none';
  }
});

uploadBtn.addEventListener('click', () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Please select an image file first.');
    return;
  }
  const formData = new FormData();
  formData.append('file', file);
  // Append the selected model to the FormData
  const selectedModel = modelSelect.value || 'mobilenet';
  formData.append('model', selectedModel);
  console.log("Selected model for file upload:", selectedModel);

  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Processing...';
  fetch('/segment', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert('Error: ' + data.error);
      } else {
        overlayImg.src = data.overlay;
        maskImg.src = data.mask;
        resultContainer.style.display = 'block';
      }
    })
    .catch(err => {
      console.error('Segmentation failed:', err);
      alert('An error occurred during segmentation.');
    })
    .finally(() => {
      uploadBtn.disabled = false;
      uploadBtn.textContent = 'Segment Image';
    });
});

// --- Live Segmentation Code ---
const liveBtn = document.getElementById('live-btn');
const video = document.getElementById('videoElement');
const liveOverlay = document.getElementById('live-overlay');
const videoContainer = document.getElementById('video-container');

modelSelect.addEventListener('change', () => {
  console.log("Model selection changed to:", modelSelect.value);
});

liveBtn.addEventListener('click', () => {
  videoContainer.style.display = 'block';
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.play();
        startLiveSegmentation();
      })
      .catch(err => console.error("Error accessing webcam:", err));
  } else {
    alert("Webcam not supported.");
  }
});

function startLiveSegmentation() {
  const captureCanvas = document.createElement('canvas');
  // Set canvas dimensions to match video (or force 512x512)
  captureCanvas.width = video.videoWidth || 512;
  captureCanvas.height = video.videoHeight || 512;
  const ctx = captureCanvas.getContext('2d');
  setInterval(() => {
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const frameData = captureCanvas.toDataURL('image/jpeg');
    const selectedModel = modelSelect.value || 'mobilenet';
    console.log("Selected model for live segmentation:", selectedModel);
    fetch('/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: frameData, model: selectedModel })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error("Segmentation error:", data.error);
      } else {
        liveOverlay.src = data.overlay;
      }
    })
    .catch(err => console.error('Live segmentation fetch error:', err));
  }, 400); // 1 frame per 0.4 second
}
