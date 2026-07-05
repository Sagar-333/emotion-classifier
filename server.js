const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process'); // This lets Node run terminal commands

const app = express();
app.use(cors());
app.use(express.json());

// The API Endpoint
app.post('/api/analyze', (req, res) => {
    const textToAnalyze = req.body.text;

    // This is exactly like typing `python predict.py "I am happy"` in the terminal
    const pythonScript = spawn('python', ['predict.py', textToAnalyze]);

    // When Python prints the answer, capture it and send it to the frontend
    pythonScript.stdout.on('data', (data) => {
        const emotion = data.toString().trim();
        res.json({ text: textToAnalyze, predicted_emotion: emotion });
    });

    // If Python crashes, catch the error
    pythonScript.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
        res.status(500).json({ error: "Failed to analyze text" });
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log(`Send a POST request to /api/analyze to test it!`);
});