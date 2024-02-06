let files = [];

document.getElementById('imageInput').addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const previewImagesContainer = document.getElementById('previewImages');
    previewImagesContainer.innerHTML = '';

    files = event.target.files;

    for (let i = 0; i < files.length && i < 12; i++) {
        const file = files[i];
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = `Image ${i + 1}`;
        previewImagesContainer.appendChild(img);
    }

    // Call the function to upload images to the server
    $("#submit").on("click", function (event) {
        event.preventDefault(); // Prevent the default form submission behavior
        if ($.isEmptyObject(files)) {
            showAlert("Please, enter all the fields")
            return false;
        }

        if (!$.isEmptyObject(files)) {
            uploadImages();
        }
    });
}

// Function to upload images to the server
function uploadImages() {
    const formData = new FormData();

    
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }

    $.ajax({
        url: '/upload', // The endpoint where you're uploading images
        type: 'POST',
        data: formData,
        contentType: false, // These two settings are important for FormData
        processData: false,
        success: function(data) {
            // Update the frontend with the received data
            $('#heartRateResult').text(data.heart_rate_bpm.toFixed(2) + ' BPM');
            $('#rhythmResult').text(data.rhythm_classification);
            $('#anatomicLocationResult').text(data.inferred_anatomic_location);
        },
        error: function(xhr, status, error) {
            // Handle any error here
            console.error("Error: ", error);
        }
    })
}
