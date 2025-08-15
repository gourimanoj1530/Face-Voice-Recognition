import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import torch
import librosa
from model import DeepFaceVoiceRecognizer
import io
import subprocess
import logging
import shutil
from pydub import AudioSegment
from pydub.utils import which

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
recognizer = None
model_loaded = False
app = FastAPI(title="Live Face & Voice Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to save captured files
CAPTURE_DIR = "captures"
AUDIO_DIR = os.path.join(CAPTURE_DIR, "audio")
IMAGE_DIR = os.path.join(CAPTURE_DIR, "images")

# Create directories if they don't exist
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Global recognizer instance
recognizer = None

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"FFmpeg found at: {ffmpeg_path}")
            return True
        else:
            logger.warning("FFmpeg not found in PATH")
            return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {e}")
        return False

def load_models():
    """Load the trained PyTorch models"""
    global recognizer, model_loaded
    
    model_path = "deep_face_voice_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found!")
        model_loaded = False
        return False
    
    try:
        recognizer = DeepFaceVoiceRecognizer()
        recognizer.load_models(model_path)
        logger.info("Models loaded successfully!")
        model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        model_loaded = False
        return False


def save_audio_file(audio_data, filename, directory=AUDIO_DIR):
    """Save audio file with proper naming convention"""
    try:
        if not filename.startswith('voice_'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            extension = filename.split('.')[-1] if '.' in filename else 'webm'
            filename = f"voice_{timestamp}.{extension}"
        
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Audio file saved: {filepath} ({len(audio_data)} bytes)")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        return None

def save_image_file(image_data, filename, directory=IMAGE_DIR):
    """Save image file with proper naming convention"""
    try:
        if not filename.startswith('face_'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            extension = filename.split('.')[-1] if '.' in filename else 'jpg'
            filename = f"face_{timestamp}.{extension}"
        
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Image file saved: {filepath} ({len(image_data)} bytes)")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image file: {e}")
        return None

def convert_audio_to_wav(input_data, input_format="webm"):
    """Convert audio data to WAV format using FFmpeg for reliability"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as tmp_in:
            tmp_in.write(input_data)
            input_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name
        
        # Convert using FFmpeg
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",          # Mono audio
            "-ar", "16000",      # 16kHz sample rate
            "-acodec", "pcm_s16le",  # 16-bit PCM
            output_path
        ]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Read converted audio
        with open(output_path, "rb") as f:
            wav_data = f.read()
        
        # Clean up temporary files
        os.unlink(input_path)
        os.unlink(output_path)
        
        return wav_data
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return None
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None

def detect_audio_format(filename, content_type):
    """Improved audio format detection"""
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    extension_map = {
        'wav': 'wav',
        'webm': 'webm',
        'mp3': 'mp3',
        'ogg': 'ogg',
        'm4a': 'm4a',
        'flac': 'flac',
        'opus': 'opus',
        'aac': 'aac'
    }
    
    if ext in extension_map:
        return extension_map[ext]
    
    content_type_map = {
        'audio/wav': 'wav',
        'audio/webm': 'webm',
        'audio/mpeg': 'mp3',
        'audio/ogg': 'ogg',
        'audio/mp4': 'm4a',
        'audio/x-m4a': 'm4a',
        'audio/flac': 'flac',
        'audio/x-flac': 'flac',
        'audio/opus': 'opus',
        'audio/aac': 'aac'
    }
    
    if content_type:
        content_type = content_type.lower()
        for mime, fmt in content_type_map.items():
            if mime in content_type:
                return fmt
    
    return 'webm'

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting up Live Face & Voice Recognition API...")
    check_ffmpeg()
    success = load_models()
    if not success:
        logger.warning("Failed to load models. Training may be required.")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "ok",
            "models_loaded": model_loaded,
            "ffmpeg_available": check_ffmpeg(),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/")
def root():
    status = "ready" if model_loaded else "models not loaded"
    return {
        "status": f"Live Face & Voice Recognition API is running - {status}",
        "endpoints": {
            "health": "/health",
            "face": "/predict-face/",
            "voice": "/predict-voice/", 
            "combined": "/predict-combined/"
        },
        "dependencies": {
            "ffmpeg_available": check_ffmpeg(),
            "models_loaded": model_loaded
        },
        "storage": {
            "capture_dir": CAPTURE_DIR,
            "audio_dir": AUDIO_DIR,
            "image_dir": IMAGE_DIR
        }
    }


@app.post("/predict-face/")
async def predict_face(file: UploadFile = File(None), request: Request = None):
    """Predict person from face image"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get image data
        if file is not None:
            image_data = await file.read()
            filename = file.filename or "uploaded_image.jpg"
        else:
            image_data = await request.body()
            filename = "captured_image.jpg"
            
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data received")
        
        # Save the image file
        saved_filepath = save_image_file(image_data, filename)
        if saved_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save image file")
        
        # Get prediction
        person, confidence = recognizer.predict_face(saved_filepath)
        
        # Get image dimensions
        try:
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_dimensions = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "unknown"
        except Exception as e:
            logger.error(f"Error getting image dimensions: {e}")
            img_dimensions = "unknown"
        
        return {
            "person": person,
            "confidence": float(confidence),
            "saved_file": saved_filepath,
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": img_dimensions
        }
        
    except Exception as e:
        logger.error(f"Face prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Face prediction failed: {str(e)}")

@app.post("/predict-voice/")
async def predict_voice(file: UploadFile = File(...)):
    """Predict person from voice audio"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        filename = file.filename or "uploaded_audio.webm"
        content_type = file.content_type or ""
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data received")

        min_audio_length = 1.0  # 1 second minimum
        try:
            # Create temporary file for validation
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            # Get audio length
            result = subprocess.run(
                ["ffprobe", "-i", tmp_path, "-show_entries", "format=duration", 
                 "-v", "quiet", "-of", "csv=p=0"],
                capture_output=True,
                text=True
            )
            duration = float(result.stdout.strip())
            
            if duration < min_audio_length:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Audio too short ({duration:.2f}s). Minimum {min_audio_length}s required."
                )
        finally:
            os.unlink(tmp_path)

        # Save original audio
        original_filepath = save_audio_file(audio_data, filename)
        if original_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save audio file")
        
        # Detect audio format
        audio_format = detect_audio_format(filename, content_type)
        logger.info(f"Detected audio format: {audio_format}")

        # Convert to WAV
        wav_data = convert_audio_to_wav(audio_data, audio_format)
        if wav_data is None:
            raise HTTPException(status_code=400, detail="Audio conversion failed")
            
        # Save converted audio
        wav_filename = filename.rsplit('.', 1)[0] + '.wav'
        wav_filepath = save_audio_file(wav_data, wav_filename)
        if wav_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save converted audio")
        
        # Get prediction
        person, confidence = recognizer.predict_voice(wav_filepath)
        
        # Get audio info
        try:
            audio_np, sr = librosa.load(io.BytesIO(wav_data), sr=16000)
            audio_info = {
                "sample_rate": sr,
                "duration_seconds": len(audio_np) / sr,
                "samples": len(audio_np)
            }
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            audio_info = None
        
        return {
            "person": person,
            "confidence": float(confidence),
            "audio_length": len(audio_np) / sr if audio_info else 0,
            "timestamp": datetime.now().isoformat(),
            "files": {
                "original": original_filepath,
                "converted": wav_filepath
            },
            "audio_info": audio_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice prediction failed: {str(e)}")

@app.post("/predict-combined/")
async def predict_combined(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """Predict person using both face and voice with more weight to face"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Save image
        img_data = await image.read()
        image_filepath = save_image_file(img_data, image.filename or "combined_image.jpg")
        if image_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save image file")
        
        # Save and convert audio
        audio_data = await audio.read()
        audio_filepath = save_audio_file(audio_data, audio.filename or "combined_audio.webm")
        if audio_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save audio file")
        
        audio_format = detect_audio_format(audio.filename, audio.content_type)
        wav_data = convert_audio_to_wav(audio_data, audio_format)
        if wav_data is None:
            raise HTTPException(status_code=400, detail="Audio conversion failed")
            
        wav_filepath = save_audio_file(wav_data, "combined_audio.wav")
        if wav_filepath is None:
            raise HTTPException(status_code=500, detail="Failed to save converted audio")
        
        # Get predictions with face weighting
        combined_person, combined_conf = recognizer.predict_combined(image_filepath, wav_filepath)
        
        # Get individual predictions for reporting
        face_person, face_conf = recognizer.predict_face(image_filepath)
        voice_person, voice_conf = recognizer.predict_voice(wav_filepath)
        
        return {
            "person": combined_person,
            "confidence": float(combined_conf),
            "individual_predictions": {
                "face": {"person": face_person, "confidence": float(face_conf)},
                "voice": {"person": voice_person, "confidence": float(voice_conf)}
            },
            "timestamp": datetime.now().isoformat(),
            "files": {
                "image": image_filepath,
                "audio_original": audio_filepath,
                "audio_converted": wav_filepath
            },
            "weighting": {
                "face": 0.7,
                "voice": 0.3,
                "combined_model": 0.5
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Combined prediction failed: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )