const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { 
  cors: { 
    origin: "*",
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"]
  } 
});
// Global variables
let mediaRecorder;
let audioChunks = [];
const RECORD_DURATION = 3000; // 3 seconds

// Start recording function
async function startRecording() {
    audioChunks = [];
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Combine all chunks
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            // Create FormData and send
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            
            // Send to server
            try {
                const response = await fetch('/predict-voice/', {
                    method: 'POST',
                    body: formData
                });
                
                // Process response...
            } catch (error) {
                console.error('Error sending audio:', error);
            }
        };
        
        mediaRecorder.start();
        
        // Stop automatically after RECORD_DURATION
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, RECORD_DURATION);
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
    }
}
// Configuration
const BACKEND_URL = 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT = 60000; // Increased timeout for deep learning models
const MAX_FILE_SIZE = '50mb';

// Middleware
app.use(express.json({ limit: MAX_FILE_SIZE }));
app.use(express.urlencoded({ limit: MAX_FILE_SIZE, extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// API Client Configuration
const apiClient = axios.create({
  baseURL: BACKEND_URL,
  timeout: REQUEST_TIMEOUT,
  maxContentLength: Infinity,
  maxBodyLength: Infinity
});

// Enhanced Error Interceptor
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.code === 'ECONNREFUSED') {
      error.message = 'Backend service unavailable. Please ensure the Python server is running.';
    } else if (error.response) {
      // Handle deep learning specific errors
      if (error.response.data.detail?.includes('CUDA')) {
        error.message = 'GPU processing error: ' + error.response.data.detail;
      } else if (error.response.status === 413) {
        error.message = 'Data too large for processing';
      }
    }
    return Promise.reject(error);
  }
);

// Helper Functions
function generateTimestampFilename(extension = 'webm') {
  const now = new Date();
  return `capture_${now.toISOString().replace(/[:.]/g, '-').replace('T', '_')}.${extension}`;
}

function validateMediaType(data, expectedType) {
  if (!data) return false;
  if (expectedType === 'image') {
    return data.startsWith('data:image/');
  } else if (expectedType === 'audio') {
    return data.startsWith('data:audio/');
  }
  return false;
}

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', async (req, res) => {
  try {
    const response = await apiClient.get('/health');
    res.json({
      status: 'healthy',
      backend: response.data,
      timestamp: new Date().toISOString(),
      model: 'deep_learning'
    });
  } catch (error) {
    res.status(503).json({
      status: 'backend_unavailable',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

app.get('/model-info', async (req, res) => {
  try {
    const [backendRes, fileCheck] = await Promise.all([
      apiClient.get('/'),
      fs.promises.access('deep_face_voice_model.pth', fs.constants.F_OK)
    ]);
    
    res.json({
      model_loaded: true,
      framework: 'pytorch',
      backend_status: backendRes.data.status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      model_loaded: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Socket.IO Handlers
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  // Heartbeat
  socket.on('ping', (cb) => cb('pong'));

  // Face Prediction
  socket.on('face_frame', async (dataURL, callback) => {
    try {
      if (!validateMediaType(dataURL, 'image')) {
        throw new Error('Invalid image data format');
      }

      const base64Data = dataURL.split(',')[1];
      const buffer = Buffer.from(base64Data, 'base64');
      const filename = generateTimestampFilename('jpg');

      const formData = new FormData();
      formData.append('file', buffer, {
        filename,
        contentType: 'image/jpeg'
      });

      const response = await apiClient.post('/predict-face/', formData, {
        headers: formData.getHeaders()
      });

      const result = {
        ...response.data,
        model_type: 'deep_learning',
        timestamp: new Date().toISOString()
      };

      callback({ status: 'success', data: result });
    } catch (error) {
      console.error('Face prediction error:', error.message);
      callback({ 
        status: 'error',
        error: error.message,
        type: 'face_prediction_error'
      });
    }
  });

  // Voice Prediction
  socket.on('voice_blob', async (audioData, metadata = {}, callback) => {
    try {
      if (!validateMediaType(audioData, 'audio')) {
        throw new Error('Invalid audio data format');
      }

      const base64Data = audioData.split(',')[1];
      const buffer = Buffer.from(base64Data, 'base64');
      const filename = metadata.filename || generateTimestampFilename('webm');

      const formData = new FormData();
      formData.append('file', buffer, {
        filename,
        contentType: metadata.contentType || 'audio/webm'
      });

      const response = await apiClient.post('/predict-voice/', formData, {
        headers: formData.getHeaders()
      });

      const result = {
        ...response.data,
        model_type: 'deep_learning',
        timestamp: new Date().toISOString()
      };

      callback({ status: 'success', data: result });
    } catch (error) {
      console.error('Voice prediction error:', error.message);
      callback({ 
        status: 'error',
        error: error.message,
        type: 'voice_prediction_error'
      });
    }
  });

  // Combined Prediction
  socket.on('combined_prediction', async ({ imageData, audioData }, callback) => {
    try {
      if (!validateMediaType(imageData, 'image') || !validateMediaType(audioData, 'audio')) {
        throw new Error('Invalid image or audio data format');
      }

      const imageBuffer = Buffer.from(imageData.split(',')[1], 'base64');
      const audioBuffer = Buffer.from(audioData.split(',')[1], 'base64');

      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: generateTimestampFilename('jpg'),
        contentType: 'image/jpeg'
      });
      formData.append('audio', audioBuffer, {
        filename: generateTimestampFilename('webm'),
        contentType: 'audio/webm'
      });

      const response = await apiClient.post('/predict-combined/', formData, {
        headers: formData.getHeaders()
      });

      const result = {
        ...response.data,
        model_type: 'deep_learning',
        timestamp: new Date().toISOString()
      };

      callback({ status: 'success', data: result });
    } catch (error) {
      console.error('Combined prediction error:', error.message);
      callback({ 
        status: 'error',
        error: error.message,
        type: 'combined_prediction_error'
      });
    }
  });

  // Disconnection
  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
  });
});

// Error Handling
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

// Server Initialization
const PORT = process.env.PORT || 3000;
server.listen(PORT, '127.0.0.1', () => {
  console.log(`🚀 Gateway server running on http://127.0.0.1:${PORT}`);
  console.log(`🔗 Connected to backend at ${BACKEND_URL}`);
  
  // Verify backend connection
  apiClient.get('/health')
    .then(() => console.log('✅ Backend connection verified'))
    .catch(() => console.warn('⚠️  Backend connection failed'));
});

// Graceful Shutdown
const shutdown = () => {
  console.log('\n🛑 Shutting down gracefully...');
  io.close(() => {
    server.close(() => {
      console.log('✅ Server closed');
      process.exit(0);
    });
  });
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);