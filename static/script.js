// Get references to DOM elements
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview');
const overlayImg = document.getElementById('overlay-img');
const maskImg = document.getElementById('mask-img');
const resultContainer = document.getElementById('result-container');
const uploadBtn = document.getElementById('upload-btn');

// Display image preview when a file is selected
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    // Show a preview of the selected image
    previewImg.src = URL.createObjectURL(file);
    previewImg.style.display = 'block';
    // Hide previous results (if any) when a new file is chosen
    resultContainer.style.display = 'none';
  }
});

// Handle the upload and segmentation process
uploadBtn.addEventListener('click', () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Please select an image file first.');
    return;
  }

  // Prepare form data
  const formData = new FormData();
  formData.append('file', file);

  // Disable the button and show processing state
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Processing...';

  // Send the file to the server via POST
  fetch('/segment', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert('Error: ' + data.error);
      } else {
        // Set the result images from the returned data URIs
        overlayImg.src = data.overlay;
        maskImg.src = data.mask;
        // Display the result container
        resultContainer.style.display = 'block';
        // Scroll to results (optional)
        resultContainer.scrollIntoView({ behavior: 'smooth' });
      }
    })
    .catch(err => {
      console.error('Segmentation failed:', err);
      alert('An error occurred during segmentation.');
    })
    .finally(() => {
      // Re-enable the button and reset text
      uploadBtn.disabled = false;
      uploadBtn.textContent = 'Segment Image';
    });
});
