import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import pickle
import random
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from peft import LoraConfig, get_peft_model, TaskType

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Set paths
DATA_ROOT = '/DATA/rl_gaming/su_wav'
VOXCELEB1_PATH = os.path.join(DATA_ROOT, 'vox1')
VOXCELEB2_PATH = os.path.join(DATA_ROOT, 'vox2')
RESULTS_PATH = '/DATA/rl_gaming/results'
VERIFICATION_FILE = os.path.join(VOXCELEB1_PATH, 'veri_test.txt')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

# Audio Processing Functions

def read_audio_file(file_path, target_sr=24000):
    try:
        # Handle special case for trial list paths
        if '/vox1_test_wav/wav/' not in file_path and 'id10' in file_path:
            # Convert from trial list format to actual file path
            parts = file_path.split('/')
            if len(parts) >= 3:
                speaker_id, session_id, file_name = parts[-3], parts[-2], parts[-1]
                file_path = os.path.join(VOXCELEB1_PATH, 'vox1_test_wav', 'wav',
                                        speaker_id, session_id, file_name)
        
        # First try with torchaudio
        if os.path.exists(file_path):
            audio, sample_rate = torchaudio.load(file_path)
            
            # Convert stereo to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                audio = resampler(audio)
            
            return audio.squeeze(0).numpy()
        else:
            print(f"File not found: {file_path}")
            return np.zeros(target_sr)  # Return silence as fallback
            
    except Exception as e:
        try:
            # Fallback to soundfile
            audio, sr = sf.read(file_path)
            if sr != target_sr:
                # Need to resample
                from librosa import resample as lr_resample
                audio = lr_resample(y=audio, orig_sr=sr, target_sr=target_sr)
            return audio
        except Exception as e2:
            print(f"Failed to load audio {file_path}: {e2}")
            return np.zeros(target_sr)  # Return silence as fallback

# Dataset Classes

class SpeakerVerificationDataset(Dataset):
    def __init__(self, trial_list_path, feature_extractor, max_length=160000):
        self.trials = self._load_trials(trial_list_path)
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def _load_trials(self, path):
        trials = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                
                label = int(parts[0])
                audio1_path = os.path.join(VOXCELEB1_PATH, 'vox1_test_wav', 'wav', parts[1])
                audio2_path = os.path.join(VOXCELEB1_PATH, 'vox1_test_wav', 'wav', parts[2])
                
                trials.append((audio1_path, audio2_path, label))
        return trials
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        audio1_path, audio2_path, label = self.trials[idx]
        
        # Load and process first audio
        waveform1 = read_audio_file(audio1_path)
        if len(waveform1) > self.max_length:
            start = random.randint(0, len(waveform1) - self.max_length)
            waveform1 = waveform1[start:start + self.max_length]
        else:
            waveform1 = np.pad(waveform1, (0, max(0, self.max_length - len(waveform1))))
        
        # Load and process second audio
        waveform2 = read_audio_file(audio2_path)
        if len(waveform2) > self.max_length:
            start = random.randint(0, len(waveform2) - self.max_length)
            waveform2 = waveform2[start:start + self.max_length]
        else:
            waveform2 = np.pad(waveform2, (0, max(0, self.max_length - len(waveform2))))
        
        # Create inputs for the model
        inputs1 = self.feature_extractor(
            waveform1, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length
        ).input_values.squeeze(0)
        
        inputs2 = self.feature_extractor(
            waveform2, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length
        ).input_values.squeeze(0)
        
        return {
            'input1': inputs1,
            'input2': inputs2,
            'label': label,
            'path1': audio1_path,
            'path2': audio2_path
        }

class SpeakerIdentificationDataset(Dataset):
    def __init__(self, root_dir, speaker_list, feature_extractor, max_length=160000):
        self.root_dir = root_dir
        self.speakers = speaker_list
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.samples = self._create_sample_list()
        
    def _create_sample_list(self):
        samples = []
        for idx, speaker_id in enumerate(tqdm(self.speakers, desc="Preparing dataset")):
            speaker_dir = os.path.join(self.root_dir, 'vox2_test_aac/aac', speaker_id)
            
            if not os.path.exists(speaker_dir):
                print(f"Speaker directory not found: {speaker_dir}")
                continue
                
            for session in os.listdir(speaker_dir):
                session_dir = os.path.join(speaker_dir, session)
                
                if not os.path.isdir(session_dir):
                    continue
                    
                for file in os.listdir(session_dir):
                    if file.endswith('.wav'):
                        file_path = os.path.join(session_dir, file)
                        samples.append((file_path, idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, speaker_idx = self.samples[idx]
        
        # Load and process audio
        waveform = read_audio_file(file_path)
        
        # Trim or pad to max_length
        if len(waveform) > self.max_length:
            start = random.randint(0, len(waveform) - self.max_length)
            waveform = waveform[start:start + self.max_length]
        else:
            waveform = np.pad(waveform, (0, max(0, self.max_length - len(waveform))))
        
        # Create input for the model
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=24000, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length
        ).input_values.squeeze(0)
        
        return {
            'input': inputs,
            'speaker_idx': speaker_idx,
            'path': file_path
        }

# Model Architecture

class AddMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features)
        weights = F.normalize(self.weight)
        
        # Calculate cosine similarity
        cos_theta = F.linear(features, weights)
        
        # Clip values to prevent numerical issues
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
        
        # Calculate arccos
        theta = torch.acos(cos_theta)
        
        # Add margin to target classes
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, labels.view(-1, 1), 1.0)
        theta = theta + self.m * target_mask
        
        # Convert back to cosine
        cos_theta_m = torch.cos(theta)
        
        # Apply scaling factor
        return self.s * cos_theta_m

class SpeakerRecognitionModel(nn.Module):
    def __init__(self, num_speakers, model_name="microsoft/wavlm-base-plus", use_lora=False):
        super(SpeakerRecognitionModel, self).__init__()
        
        # Load pre-trained model
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.use_lora = use_lora
        
        if use_lora:
            # Apply LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,  # Rank of LoRA matrices
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["k_proj", "q_proj", "v_proj", "o_proj"]
            )
            self.wavlm = get_peft_model(self.wavlm, peft_config)
            print("Using LoRA for fine-tuning")
            self.wavlm.print_trainable_parameters()
            
            # Store reference to base model for direct access
            self.base_model = self.wavlm.base_model.model
        else:
            # Freeze weights when not using LoRA
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.base_model = None
        
        # Hidden dimensions
        hidden_size = self.wavlm.config.hidden_size
        
        # Speaker embedding projector
        self.embedding_projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Arc margin product for classification
        self.arc_margin = AddMarginProduct(256, num_speakers)
    
    def forward(self, x, labels=None, extract_embedding=False):
        # Process through WavLM
        if self.use_lora:
            # When using LoRA, access the base model directly
            outputs = self.base_model(x).last_hidden_state
        else:
            # Standard forward pass
            outputs = self.wavlm(x).last_hidden_state
        
        # Average pooling across time dimension
        pooled = torch.mean(outputs, dim=1)
        
        # Project to speaker embedding space
        embeddings = self.embedding_projector(pooled)
        
        if extract_embedding:
            return embeddings
            
        # Apply margin and get logits
        if labels is not None:
            logits = self.arc_margin(embeddings, labels)
        else:
            # During inference without labels
            logits = F.linear(F.normalize(embeddings), F.normalize(self.arc_margin.weight))
            logits = logits * self.arc_margin.s
            
        return logits, embeddings

# Evaluation Metrics

def calculate_equal_error_rate(scores, labels):
    # Sort scores and labels
    indices = np.argsort(scores)
    sorted_scores = np.array(scores)[indices]
    sorted_labels = np.array(labels)[indices]
    
    # Count positive and negative samples
    num_pos = sum(sorted_labels)
    num_neg = len(sorted_labels) - num_pos
    
    # Calculate FAR (False Accept Rate) and FRR (False Reject Rate)
    far = np.cumsum(sorted_labels) / num_pos
    frr = 1 - np.cumsum(1 - sorted_labels) / num_neg
    
    # Find where FAR equals FRR (or closest point)
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    
    return eer * 100  # Return as percentage

def calculate_tar_at_far(scores, labels, target_far=0.01):
    # Get ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find point with FAR closest to target
    idx = np.argmin(np.abs(fpr - target_far))
    
    return tpr[idx]

def evaluate_verification(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating verification"):
            # Move inputs to device
            input1 = batch['input1'].to(device)
            input2 = batch['input2'].to(device)
            labels = batch['label'].numpy()
            
            # Extract embeddings
            embedding1 = model(input1, extract_embedding=True)
            embedding2 = model(input2, extract_embedding=True)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2).cpu().numpy()
            
            all_scores.extend(similarity)
            all_labels.extend(labels)
    
    # Calculate metrics
    eer = calculate_equal_error_rate(all_scores, all_labels)
    tar_at_far = calculate_tar_at_far(all_scores, all_labels, 0.01)
    
    return eer, tar_at_far, all_scores, all_labels

def evaluate_identification(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move inputs to device
            inputs = batch['input'].to(device)
            labels = batch['speaker_idx'].numpy()
            
            # Extract embeddings
            embeddings = model(inputs, extract_embedding=True)
            
            all_embeddings.extend(embeddings.cpu().numpy())
            all_labels.extend(labels)
    
    # Normalize embeddings
    all_embeddings = normalize(np.array(all_embeddings))
    all_labels = np.array(all_labels)
    
    # Perform identification
    unique_labels = np.unique(all_labels)
    
    # Use first occurrence of each speaker as enrollment
    enrollment_embeddings = []
    enrollment_labels = []
    
    for label in unique_labels:
        idx = np.where(all_labels == label)[0][0]
        enrollment_embeddings.append(all_embeddings[idx])
        enrollment_labels.append(label)
    
    enrollment_embeddings = np.array(enrollment_embeddings)
    
    # Test on remaining samples
    correct = 0
    total = 0
    
    for i, embedding in enumerate(all_embeddings):
        # Skip enrollment samples
        if i in np.where(all_labels == all_labels[i])[0][:1]:
            continue
            
        # Compute similarities with enrollment embeddings
        similarities = np.dot(enrollment_embeddings, embedding)
        predicted_idx = np.argmax(similarities)
        predicted_label = enrollment_labels[predicted_idx]
        
        if predicted_label == all_labels[i]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

# Training Functions

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-4, device=DEVICE):
    print(f"Training model for {epochs} epochs on {device}")
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            # Get data
            inputs = batch['input'].to(device)
            speaker_idx = batch['speaker_idx'].to(device)
            
            # Forward pass
            logits, _ = model(inputs, speaker_idx)
            loss = criterion(logits, speaker_idx)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                # Get data
                inputs = batch['input'].to(device)
                speaker_idx = batch['speaker_idx'].to(device)
                
                # Forward pass
                logits, _ = model(inputs, speaker_idx)
                loss = criterion(logits, speaker_idx)
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += speaker_idx.size(0)
                correct += (predicted == speaker_idx).sum().item()
                
                val_loss += loss.item()
        
        # Calculate metrics
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_PATH, 'best_model.pt'))
            print(f"Saved best model with accuracy {best_acc:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_model.pt')))
    return model

def select_speakers(root_dir, num_train=100, num_test=18):
    speaker_dir = os.path.join(root_dir, 'vox2_test_aac/aac')
    
    # Get all speaker directories
    all_speakers = sorted([d for d in os.listdir(speaker_dir) 
                           if os.path.isdir(os.path.join(speaker_dir, d))])
    
    # Split into train and test sets
    train_speakers = all_speakers[:num_train]
    test_speakers = all_speakers[num_train:num_train + num_test]
    
    return train_speakers, test_speakers

def plot_roc_curves(pretrained_scores, pretrained_labels, finetuned_scores, finetuned_labels, pretrained_eer, finetuned_eer):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve for pretrained model
    fpr_pre, tpr_pre, _ = roc_curve(pretrained_labels, pretrained_scores)
    roc_auc_pre = auc(fpr_pre, tpr_pre)
    
    # Calculate ROC curve for finetuned model
    fpr_ft, tpr_ft, _ = roc_curve(finetuned_labels, finetuned_scores)
    roc_auc_ft = auc(fpr_ft, tpr_ft)
    
    # Plot curves
    plt.plot(fpr_pre, tpr_pre, color='blue', lw=2, 
             label=f'Pretrained (EER={pretrained_eer:.2f}%, AUC={roc_auc_pre:.3f})')
    plt.plot(fpr_ft, tpr_ft, color='red', lw=2, 
             label=f'Fine-tuned (EER={finetuned_eer:.2f}%, AUC={roc_auc_ft:.3f})')
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Speaker Verification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(RESULTS_PATH, 'roc_comparison.png'), dpi=300)
    plt.close()

def main():
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Get speaker lists for training and testing
    train_speakers, test_speakers = select_speakers(VOXCELEB2_PATH)
    print(f"Selected {len(train_speakers)} speakers for training and {len(test_speakers)} speakers for testing")
    
    # Initialize verification dataset (for evaluation only)
    verification_dataset = SpeakerVerificationDataset(VERIFICATION_FILE, feature_extractor)
    verification_loader = DataLoader(verification_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize identification datasets
    train_dataset = SpeakerIdentificationDataset(VOXCELEB2_PATH, train_speakers, feature_extractor)
    test_dataset = SpeakerIdentificationDataset(VOXCELEB2_PATH, test_speakers, feature_extractor)
    
    print(f"Created datasets: {len(train_dataset)} training samples, {len(test_dataset)} testing samples, {len(verification_dataset)} verification pairs")
    
    # Save feature extractor for later use
    with open(os.path.join(RESULTS_PATH, 'feature_extractor.pkl'), 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # STEP 1: Evaluate pretrained model
    if os.path.exists(os.path.join(RESULTS_PATH, 'pretrained_results.pkl')):
        # Load cached results
        print("Loading cached pretrained model results...")
        with open(os.path.join(RESULTS_PATH, 'pretrained_results.pkl'), 'rb') as f:
            pretrained_results = pickle.load(f)
        pretrained_eer = pretrained_results['eer']
        pretrained_tar = pretrained_results['tar_at_far']
        pretrained_scores = pretrained_results['scores']
        pretrained_labels = pretrained_results['labels']
        pretrained_ident_acc = pretrained_results['ident_acc']
    else:
        print("Evaluating pretrained model...")
        # Initialize pretrained model
        pretrained_model = SpeakerRecognitionModel(
            num_speakers=len(train_speakers),
            model_name="microsoft/wavlm-base-plus",
            use_lora=False
        ).to(DEVICE)
        
        # Save pretrained model
        torch.save(pretrained_model.state_dict(), os.path.join(RESULTS_PATH, 'pretrained_model.pt'))
        
        # Evaluate on verification task
        pretrained_eer, pretrained_tar, pretrained_scores, pretrained_labels = evaluate_verification(
            pretrained_model, verification_loader, DEVICE
        )
        
        # Evaluate on identification task
        pretrained_ident_acc = evaluate_identification(pretrained_model, DataLoader(test_dataset, batch_size=8), DEVICE)
        
        # Cache results
        pretrained_results = {
            'eer': pretrained_eer,
            'tar_at_far': pretrained_tar,
            'scores': pretrained_scores,
            'labels': pretrained_labels,
            'ident_acc': pretrained_ident_acc
        }
        with open(os.path.join(RESULTS_PATH, 'pretrained_results.pkl'), 'wb') as f:
            pickle.dump(pretrained_results, f)
    
    print(f"Pretrained model performance:")
    print(f"  Equal Error Rate: {pretrained_eer:.2f}%")
    print(f"  TAR@1%FAR: {pretrained_tar:.4f}")
    print(f"  Identification Accuracy: {pretrained_ident_acc:.4f}")
    
    # STEP 2: Fine-tune model
    if os.path.exists(os.path.join(RESULTS_PATH, 'best_model.pt')):
        print("Loading previously fine-tuned model...")
        finetuned_model = SpeakerRecognitionModel(
            num_speakers=len(train_speakers),
            model_name="microsoft/wavlm-base-plus",
            use_lora=True
        ).to(DEVICE)
        finetuned_model.load_state_dict(torch.load(os.path.join(RESULTS_PATH, 'best_model.pt')))
    else:
        print("Fine-tuning model...")
        # Initialize model with LoRA
        finetuned_model = SpeakerRecognitionModel(
            num_speakers=len(train_speakers),
            model_name="microsoft/wavlm-base-plus",
            use_lora=True
        ).to(DEVICE)
        
        # Train model
        finetuned_model = train_model(
            finetuned_model,
            train_loader,
            DataLoader(test_dataset, batch_size=8),
            epochs=3,
            lr=1e-4
        )
    
    # Save fine-tuned model
    torch.save(finetuned_model.state_dict(), os.path.join(RESULTS_PATH, 'finetuned_model.pt'))
    
    # STEP 3: Evaluate fine-tuned model
    if os.path.exists(os.path.join(RESULTS_PATH, 'finetuned_results.pkl')):
        # Load cached results
        print("Loading cached fine-tuned model results...")
        with open(os.path.join(RESULTS_PATH, 'finetuned_results.pkl'), 'rb') as f:
            finetuned_results = pickle.load(f)
        finetuned_eer = finetuned_results['eer']
        finetuned_tar = finetuned_results['tar_at_far']
        finetuned_scores = finetuned_results['scores']
        finetuned_labels = finetuned_results['labels']
        finetuned_ident_acc = finetuned_results['ident_acc']
    else:
        print("Evaluating fine-tuned model...")
        # Evaluate on verification task
        finetuned_eer, finetuned_tar, finetuned_scores, finetuned_labels = evaluate_verification(
            finetuned_model, verification_loader, DEVICE
        )
        
        # Evaluate on identification task
        finetuned_ident_acc = evaluate_identification(finetuned_model, DataLoader(test_dataset, batch_size=8), DEVICE)
        
        # Cache results
        finetuned_results = {
            'eer': finetuned_eer,
            'tar_at_far': finetuned_tar,
            'scores': finetuned_scores,
            'labels': finetuned_labels,
            'ident_acc': finetuned_ident_acc
        }
        with open(os.path.join(RESULTS_PATH, 'finetuned_results.pkl'), 'wb') as f:
            pickle.dump(finetuned_results, f)
    
    print(f"Fine-tuned model performance:")
    print(f"  Equal Error Rate: {finetuned_eer:.2f}%")
    print(f"  TAR@1%FAR: {finetuned_tar:.4f}")
    print(f"  Identification Accuracy: {finetuned_ident_acc:.4f}")
    
    # STEP 4: Plot comparison results
    plot_roc_curves(
        pretrained_scores, pretrained_labels,
        finetuned_scores, finetuned_labels,
        pretrained_eer, finetuned_eer
    )
    
    # Save comparison results to CSV
    results_data = {
        'Model': ['Pretrained', 'Fine-tuned'],
        'EER (%)': [pretrained_eer, finetuned_eer],
        'TAR@1%FAR': [pretrained_tar, finetuned_tar],
        'Identification Accuracy': [pretrained_ident_acc, finetuned_ident_acc]
    }
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(RESULTS_PATH, 'comparison_results.csv'), index=False)
    
    print(f"Results saved to {os.path.join(RESULTS_PATH, 'comparison_results.csv')}")
    print(f"ROC curves saved to {os.path.join(RESULTS_PATH, 'roc_comparison.png')}")

if __name__ == "__main__":
    main()