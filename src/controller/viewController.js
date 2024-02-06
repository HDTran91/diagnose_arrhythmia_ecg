import multer from 'multer'
import { spawn } from 'child_process';

const fs = require('fs');
const path = require('path');


let mainPage = (req, res) => {
    return res.render("home");
}

let uploadImages = (req, res) => {
    // Access the uploaded files from req.files
    const images = req.files;

    // Save uploaded images to disk
    const imagePaths = [];
    Object.values(images).forEach((file, index) => {
    // Check the correct property (e.g., file.buffer)
    const fileData = file.buffer || file.data;

    // Save images to the 'uploads' folder
    const fileName = `image_${index}.jpg`;
    const filePath = path.join(__dirname, './../uploads', fileName);
    fs.writeFileSync(filePath, fileData);
    imagePaths.push(fileName); // Push only the filename, not the full path
});
    // console.log(imagePaths)
    // Spawn a Python process and send paths to the Python script
    const pythonProcess = spawn('python', ['process_images.py', JSON.stringify(imagePaths)]);

    // Listen for Python script output
    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python script output: ${data}`);
    });

    // Respond with a success message
    res.status(200).json({ message: 'Images received and temporarily stored.' });
};

module.exports = {
    mainPage: mainPage,
    uploadImages: uploadImages
};
