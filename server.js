const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process'); // This lets Node run terminal commands

const app = express();
app.use(cors());
app.use(express.json());

// The API Endpoint
app.post('/api/analyze', (req, res) => {
    const textToAnalyze = req.body.text;
    const pythonScript = spawn('python3', ['predict.py', textToAnalyze]);

    let result = '';

    pythonScript.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonScript.stderr.on('data', (data) => {
        console.error(`Python stderr (may just be a warning): ${data}`);
    });

    pythonScript.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: "Failed to analyze text" });
        }
        res.json({ text: textToAnalyze, predicted_emotion: result.trim() });
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log(`Send a POST request to /api/analyze to test it!`);
});