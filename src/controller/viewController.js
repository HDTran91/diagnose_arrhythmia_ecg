const fs = require('fs');
const path = require('path');
const multer = require('multer'); // Assuming multer is already set up correctly elsewhere
const { spawn } = require('child_process');

let mainPage = (req, res) => {
    return res.render("home");
};

const uploadFiles = (req, res) => {
    const files = req.files;
    const filePaths = [];

    // Ensure the uploads directory exists
    const uploadsDir = path.join(__dirname, './../uploads');
    if (!fs.existsSync(uploadsDir)){
        fs.mkdirSync(uploadsDir, { recursive: true });
    }

    files.forEach((file, index) => {
        const fileData = file.buffer;
        const filePath = path.join(uploadsDir, file.originalname);
        fs.writeFileSync(filePath, fileData);
        filePaths.push(filePath);
    });

    // Make sure to use the '-u' flag for unbuffered output
    const pythonProcess = spawn('python', ['-u', 'process_files.py', JSON.stringify(filePaths)]);
    let fullOutput = '';

    pythonProcess.stdout.on('data', (data) => {
        fullOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        // Check if we received a complete JSON string
        const jsonStart = fullOutput.indexOf('---JSON_START---') + '---JSON_START---'.length;
        const jsonEnd = fullOutput.indexOf('---JSON_END---');
        if (jsonStart >= 0 && jsonEnd > jsonStart) {
            const jsonString = fullOutput.substring(jsonStart, jsonEnd).trim();
            try {
                const outputFromPython = JSON.parse(jsonString);
                console.log("Python script output:", outputFromPython);
                res.status(200).json(outputFromPython);
            } catch (error) {
                console.error("Error parsing JSON from Python script:", error);
                res.status(500).json({ message: "Error processing files" });
            }
        } else {
            // If no JSON was found, respond with a generic error or the raw output for debugging
            res.status(500).json({ message: "Error processing files, no output from Python script", rawOutput: fullOutput });
        }
    });
};

module.exports = {
    mainPage: mainPage,
    uploadFiles: uploadFiles
};
