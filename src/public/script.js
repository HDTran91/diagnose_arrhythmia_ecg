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
        url: "/upload",
        type: "POST",
        data: formData,
        contentType: false,
        processData: false,
        success: function(data){
            console.log(data);
        },
        error: function(error){
            console.error('Error:', error);
        }
    });
}
