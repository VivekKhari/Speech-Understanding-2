# Required Libraries
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Task A: MFCC Extraction and Statistical Analysis

# Parameters
n_mfcc = 13  
languages = ['Hindi', 'Punjabi', 'Marathi']  
base_dir = '.' 
num_samples = 2500  # 2500 samples per language

def extract_mfcc_features(file_path, n_mfcc=13):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_var = np.var(mfccs, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_var))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features and labels
data = []
labels = []
for lang in languages:
    lang_dir = os.path.join(base_dir, lang)
    print(f"Processing {lang}...")
    count = 0
    for file_name in os.listdir(lang_dir):
        if count >= num_samples:
            break
        file_path = os.path.join(lang_dir, file_name)
        features = extract_mfcc_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(lang)
            count += 1
    print(f"Completed {count}/{num_samples} files for {lang}")

# Create DataFrame
columns = [f'MFCC{i+1}_mean' for i in range(n_mfcc)] + [f'MFCC{i+1}_var' for i in range(n_mfcc)]
df = pd.DataFrame(data, columns=columns)
df['label'] = labels
df.to_csv('mfcc_features.csv', index=False)


# Visualization of MFCC Spectrograms (3 per language)

plt.figure(figsize=(15, 10))
for lang_idx, lang in enumerate(languages):
    lang_dir = os.path.join(base_dir, lang)
    files = os.listdir(lang_dir)[:3]  # First 3 files
    
    for file_idx, file_name in enumerate(files):
        file_path = os.path.join(lang_dir, file_name)
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        plt.subplot(3, 3, lang_idx * 3 + file_idx + 1)
        librosa.display.specshow(mfccs, x_axis='time', sr=sr)
        plt.title(f'{lang} - Sample {file_idx+1}')
        plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.savefig('mfcc_spectrograms.png')
plt.close()  # Close the figure to free memory

# Statistical Analysis (Mean and Variance Plots)

# Compute mean and variance of coefficients
stats_mean = df.groupby('label').mean().reset_index()
stats_var = df.groupby('label').var().reset_index()

# Plot mean of MFCC coefficients
plt.figure(figsize=(12, 6))
for lang in languages:
    lang_data = stats_mean[stats_mean['label'] == lang]
    plt.plot(lang_data.iloc[0, 1:n_mfcc+1], label=lang, marker='o')
plt.title('Mean of MFCC Coefficients (Across Languages)')
plt.xlabel('MFCC Index')
plt.ylabel('Mean Value')
plt.legend()
plt.grid(True)
plt.savefig('mfcc_mean_comparison.png')
plt.close()

# Plot variance of MFCC coefficients
plt.figure(figsize=(12, 6))
for lang in languages:
    lang_data = stats_var[stats_var['label'] == lang]
    plt.plot(lang_data.iloc[0, 1:n_mfcc+1], label=lang, marker='o')
plt.title('Variance of MFCC Coefficients (Across Languages)')
plt.xlabel('MFCC Index')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.savefig('mfcc_var_comparison.png')
plt.close()


# Task B: Classification with Confusion Matrix

X = df.drop('label', axis=1).values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=languages)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=languages)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=languages))