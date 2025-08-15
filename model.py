import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import os
import numpy as np
from PIL import Image
import glob
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import librosa
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepFaceVoiceRecognizer:
    def __init__(self):
        self.people = ['Edwin', 'Gouri', 'Jessica', 'Dale']
        self.num_classes = len(self.people)
        self.label_encoder = {p: i for i, p in enumerate(self.people)}
        self.reverse_encoder = {i: p for i, p in enumerate(self.people)}
        
        # Initialize models
        self.face_model = self._init_face_model()
        self.voice_model = self._init_voice_model()
        self.combined_model = None  # Will be initialized when loading or training
        
        # Store dimensions for combined model
        self.face_feature_dim = None
        self.voice_feature_dim = None
        
        # Transforms
        self.face_transform = self._get_face_transform()
        self.voice_transform = self._get_voice_transform()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def _init_face_model(self):
        """Initialize EfficientNet for face recognition"""
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Modify classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes))
        return model.to(device)
    
    def _init_voice_model(self):
        """Initialize CNN for voice recognition"""
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes))
        return model.to(device)
    
    def _init_combined_model(self, face_dim, voice_dim):
        """Initialize multimodal fusion model with proper dimensions"""
        self.face_feature_dim = face_dim
        self.voice_feature_dim = voice_dim
        
        model = nn.Sequential(
            nn.Linear(face_dim + voice_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        ).to(device)
        
        self.combined_model = model
        return model
    
    def _get_face_transform(self):
        """Augmentation for face images"""
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _get_voice_transform(self):
        """Transform for audio to spectrogram"""
        return T.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
    
    def detect_face(self, image_path):
        """Detect and preprocess face region"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            face_region = cv2.resize(gray, (224, 224))
        else:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            face_region = gray[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (224, 224))
            
        return Image.fromarray(face_region).convert('RGB')
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Trim/pad to 3 seconds
        target_length = 16000 * 3
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_length]
            
        return waveform
    
    def load_person_data(self, person_name, face_dir, voice_dir):
        """Load and preprocess data for one person"""
        face_data, voice_data = [], []
        pid = self.label_encoder[person_name]
        
        # Load face images
        if os.path.exists(face_dir):
            for img_file in glob.glob(os.path.join(face_dir, '*.*')):
                if img_file.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
                    face_img = self.detect_face(img_file)
                    if face_img:
                        face_data.append((self.face_transform(face_img), pid))
        
        # Load voice samples
        if os.path.exists(voice_dir):
            for audio_file in glob.glob(os.path.join(voice_dir, '*.*')):
                if audio_file.split('.')[-1].lower() in ['wav', 'mp3', 'flac']:
                    waveform = self.preprocess_audio(audio_file)
                    spectrogram = self.voice_transform(waveform)
                    voice_data.append((spectrogram, pid))
                    
        return face_data, voice_data
    
    def train_models(self, person_dirs):
        """Train all models end-to-end"""
        all_face_data = []
        all_voice_data = []
        
        # Load data for all persons
        for person, (face_dir, voice_dir) in zip(self.people, person_dirs):
            face_data, voice_data = self.load_person_data(person, face_dir, voice_dir)
            all_face_data.extend(face_data)
            all_voice_data.extend(voice_data)
        
        # Create data loaders
        face_loader = DataLoader(all_face_data, batch_size=8, shuffle=True)
        voice_loader = DataLoader(all_voice_data, batch_size=8, shuffle=True)
        
        # Train face model
        print("Training face model...")
        self._train_model(self.face_model, face_loader, epochs=50)
        
        # Train voice model
        print("\nTraining voice model...")
        self._train_model(self.voice_model, voice_loader, epochs=100)
        
        # Train combined model
        print("\nTraining combined model...")
        self._train_combined_model(all_face_data, all_voice_data)
        
        print("\nAll models trained successfully!")
    
    def _train_model(self, model, loader, epochs=30):
        """Generic training function for single models"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    def _train_combined_model(self, face_data, voice_data):
        """Train multimodal fusion model"""
        # Create aligned dataset
        combined_data = []
        
        # Get minimum count per person to ensure balanced data
        person_counts = {p: 0 for p in self.people}
        for _, label in face_data:
            person_name = self.reverse_encoder[label]
            person_counts[person_name] += 1
        
        voice_counts = {p: 0 for p in self.people}
        for _, label in voice_data:
            person_name = self.reverse_encoder[label]
            voice_counts[person_name] += 1
        
        # Use minimum samples per person
        min_samples_per_person = min(min(person_counts.values()), min(voice_counts.values()))
        
        # Create balanced dataset
        person_face_data = {p: [] for p in self.people}
        person_voice_data = {p: [] for p in self.people}
        
        for face_tensor, label in face_data:
            person_name = self.reverse_encoder[label]
            if len(person_face_data[person_name]) < min_samples_per_person:
                person_face_data[person_name].append((face_tensor, label))
        
        for voice_tensor, label in voice_data:
            person_name = self.reverse_encoder[label]
            if len(person_voice_data[person_name]) < min_samples_per_person:
                person_voice_data[person_name].append((voice_tensor, label))
        
        # Extract features and create combined dataset
        print(f"Creating combined dataset with {min_samples_per_person} samples per person...")
        
        # First, determine the actual feature dimensions
        sample_face_tensor = person_face_data[self.people[0]][0][0]
        sample_voice_tensor = person_voice_data[self.people[0]][0][0]
        
        with torch.no_grad():
            # Get face feature dimension
            face_input = sample_face_tensor.unsqueeze(0).to(device)
            face_feat = self.face_model.features(face_input)
            face_feat = self.face_model.avgpool(face_feat)
            face_feat = torch.flatten(face_feat, 1)
            face_dim = face_feat.shape[1]
            
            # Get voice feature dimension
            voice_input = sample_voice_tensor.unsqueeze(0).to(device)
            voice_feat = self.voice_model[:-2](voice_input)
            voice_dim = voice_feat.shape[1]
        
        print(f"Face feature dimension: {face_dim}")
        print(f"Voice feature dimension: {voice_dim}")
        
        # Now initialize the combined model with correct dimensions
        self._init_combined_model(face_dim, voice_dim)
        
        for person in self.people:
            face_samples = person_face_data[person]
            voice_samples = person_voice_data[person]
            
            for i in range(min(len(face_samples), len(voice_samples))):
                face_tensor, face_label = face_samples[i]
                voice_tensor, voice_label = voice_samples[i]
                
                # Extract features from trained models
                with torch.no_grad():
                    # Face features
                    face_input = face_tensor.unsqueeze(0).to(device)
                    face_feat = self.face_model.features(face_input)
                    face_feat = self.face_model.avgpool(face_feat)
                    face_feat = torch.flatten(face_feat, 1)
                    
                    # Voice features - Fixed: Add batch dimension
                    voice_input = voice_tensor.unsqueeze(0).to(device)
                    voice_feat = self.voice_model[:-2](voice_input)  # Remove last 2 layers (ReLU + Linear)
                
                # Combine features
                combined_feat = torch.cat((face_feat, voice_feat), dim=1)
                combined_data.append((combined_feat.squeeze(0), face_label))
        
        print(f"Combined dataset created with {len(combined_data)} samples")
        
        # Train combined classifier
        if len(combined_data) > 0:
            loader = DataLoader(combined_data, batch_size=4, shuffle=True)
            self._train_model(self.combined_model, loader, epochs=30)
        else:
            print("Warning: No combined data available for training")
    
    def save_models(self, path):
        """Save all models and metadata"""
        state = {
            'face_state': self.face_model.state_dict(),
            'voice_state': self.voice_model.state_dict(),
            'combined_state': self.combined_model.state_dict() if self.combined_model is not None else None,
            'face_feature_dim': self.face_feature_dim,
            'voice_feature_dim': self.voice_feature_dim,
            'label_encoder': self.label_encoder,
            'people': self.people
        }
        torch.save(state, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path):
        """Load trained models"""
        state = torch.load(path, map_location=device)
        
        # Load basic models
        self.face_model.load_state_dict(state['face_state'])
        self.voice_model.load_state_dict(state['voice_state'])
        
        # Load combined model if it exists
        if state.get('combined_state') is not None:
            face_dim = state.get('face_feature_dim')
            voice_dim = state.get('voice_feature_dim')
            
            if face_dim is not None and voice_dim is not None:
                self._init_combined_model(face_dim, voice_dim)
                self.combined_model.load_state_dict(state['combined_state'])
            else:
                print("Warning: Combined model dimensions not found in saved state")
        else:
            print("Warning: No combined model found in saved state")
        
        # Load metadata
        self.label_encoder = state['label_encoder']
        self.people = state['people']
        self.reverse_encoder = {v:k for k,v in self.label_encoder.items()}
        print(f"Models loaded from {path}")
    
    def predict_face(self, image_path):
        """Predict from face image"""
        self.face_model.eval()
        face_img = self.detect_face(image_path)
        if face_img is None:
            return "Unknown", 0.0
            
        input_tensor = self.face_transform(face_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.face_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        return self.reverse_encoder[pred_idx.item()], conf.item()
    
    def predict_voice(self, audio_path):
        """Predict from voice audio"""
        self.voice_model.eval()
        waveform = self.preprocess_audio(audio_path)
        spectrogram = self.voice_transform(waveform).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.voice_model(spectrogram)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        return self.reverse_encoder[pred_idx.item()], conf.item()
    
    def predict_combined(self, image_path, audio_path, face_weight=0.7, voice_weight=0.3):
        if self.combined_model is None:
            print("Combined model not trained yet!")
            return "Unknown", 0.0
            
        # Set all models to evaluation mode
        self.face_model.eval()
        self.voice_model.eval()
        self.combined_model.eval()
        
        # Get face features
        face_img = self.detect_face(image_path)
        if face_img is None:
            return "Unknown", 0.0
            
        face_tensor = self.face_transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            face_feat = self.face_model.features(face_tensor)
            face_feat = self.face_model.avgpool(face_feat)
            face_feat = torch.flatten(face_feat, 1)
            
            # Get face prediction separately
            face_output = self.face_model.classifier(face_feat)
            face_probs = torch.softmax(face_output, dim=1)
        
        # Get voice features
        waveform = self.preprocess_audio(audio_path)
        voice_tensor = self.voice_transform(waveform).unsqueeze(0).to(device)
        with torch.no_grad():
            voice_feat = self.voice_model[:-2](voice_tensor)
            
            # Get voice prediction separately
            voice_output = self.voice_model[-2:](voice_feat)  # Only classifier layers
            voice_probs = torch.softmax(voice_output, dim=1)
        
        # Combine features and get combined prediction
        combined = torch.cat((face_feat, voice_feat), dim=1)
        with torch.no_grad():
            combined_output = self.combined_model(combined)
            combined_probs = torch.softmax(combined_output, dim=1)
        
        # Create weighted average with more weight to face prediction
        weighted_probs = (face_weight * face_probs + 
                        voice_weight * voice_probs + 
                        0.5 * combined_probs) / (face_weight + voice_weight + 0.5)
        
        conf, pred_idx = torch.max(weighted_probs, 1)
        
        return self.reverse_encoder[pred_idx.item()], conf.item()


def main():
    """Main training function"""
    recognizer = DeepFaceVoiceRecognizer()
    
    # Directory setup - same as your original structure
    person_dirs = [
        ("data/Edwin/Faces", "data/Edwin/Voices"),
        ("data/Gouri/Faces", "data/Gouri/Voices"),
        ("data/Jessica/Faces", "data/Jessica/Voices"),
        ("data/Dale/Faces", "data/Dale/Voices")
    ]
    
    # Verify directories
    for face_dir, voice_dir in person_dirs:
        if not os.path.exists(face_dir):
            print(f"Missing face directory: {face_dir}")
            return
        if not os.path.exists(voice_dir):
            print(f"Missing voice directory: {voice_dir}")
            return
    
    # Train and save
    recognizer.train_models(person_dirs)
    recognizer.save_models("deep_face_voice_model.pth")
    print("Training completed!")


if __name__ == "__main__":
    main()