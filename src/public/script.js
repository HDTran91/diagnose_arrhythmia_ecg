let files = [];

document.getElementById('fileInput').addEventListener('change', handleFileUpload);
// Adjust preview for general files (for images, show thumbnails; for others, show file names)
function handleFileUpload(event) {
    const previewFilesContainer = document.getElementById('previewFiles');
    previewFilesContainer.innerHTML = '';

    files = event.target.files;

    for (let i = 0; i < files.length && i < 12; i++) {
        const file = files[i];

        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = `File ${i + 1}`;
            img.style.maxWidth = '100px';
            img.style.maxHeight = '100px';
            previewFilesContainer.appendChild(img);
        } else {
            const fileNameDiv = document.createElement('div');
            fileNameDiv.textContent = `File ${i + 1}: ${file.name}`;
            previewFilesContainer.appendChild(fileNameDiv);
        }
    }
}

// Separated event listener for the submit button to avoid re-binding
$("#submit").on("click", function (event) {
    event.preventDefault(); // Prevent the default form submission behavior
    if (files.length === 0) {
        alert("Please select files to upload.");
        return false;
    }

    uploadFiles();
});

// Function to upload files to the server
function uploadFiles() {
    const formData = new FormData();
    
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(data) {
            // Assuming 'data' contains the response with specific properties
      
            $('#heartRateResult').text(data.heart_rate_bpm.toFixed(2) + ' BPM');
            $('#rhythmResult').text(data.rhythm_classification);
            $('#anatomicLocationResult').text(data.inferred_anatomic_location);
            $('#ecgImage').attr('src', data.ecg_image_url);
            $('#ecgImageContainer').show();//show image
            $('#previewFiles').hide(); // Hide file input and submit
        },
        error: function(xhr, status, error) {
            console.error("Error: ", error);
            alert("Error uploading files.");
        }
    });
}
