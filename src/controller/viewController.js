import multer from 'multer'
import { spawn } from 'child_process';

const fs = require('fs');
const path = require('path');


let mainPage = (req, res) => {
    return res.render("home");
}

let uploadImages = (req, res) => {
    const images = req.files;
    const imagePaths = [];

    Object.values(images).forEach((file, index) => {
        const fileData = file.buffer || file.data;
        const fileName = `image_${index}.jpg`;
        const filePath = path.join(__dirname, './../uploads', fileName);
        fs.writeFileSync(filePath, fileData);
        imagePaths.push(fileName);
    });

    const pythonProcess = spawn('python', ['process_images.py', JSON.stringify(imagePaths)]);
    let fullOutput = '';

    pythonProcess.stdout.on('data', (data) => {
        fullOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
        const jsonStart = fullOutput.indexOf('---JSON_START---') + '---JSON_START---'.length;
        const jsonEnd = fullOutput.indexOf('---JSON_END---');
        if (jsonStart >= 0 && jsonEnd > jsonStart) {
            const jsonString = fullOutput.substring(jsonStart, jsonEnd).trim();
            try {
                const outputFromPython = JSON.parse(jsonString);
                console.log("Python script output:", outputFromPython);
                res.status(200).json(outputFromPython); // Correct placement
            } catch (error) {
                console.error("Error parsing JSON from Python script:", error);
                res.status(500).json({ message: "Error processing images" });
            }
        } else {
            // If no JSON data found, still need to handle it gracefully
            res.status(500).json({ message: "Error processing images, no output from Python script" });
        }
    });
    
    // Removed the premature response here
};

module.exports = {
    mainPage: mainPage,
    uploadImages: uploadImages
};
