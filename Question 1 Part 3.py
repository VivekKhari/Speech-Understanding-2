import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import argparse
from pathlib import Path

# Define paths
OUTPUT_ROOT = '/DATA/rl_gaming/results_iii'
MIXTURES_DIR = os.path.join(OUTPUT_ROOT, 'mixtures')
SEPARATED_DIR = os.path.join(OUTPUT_ROOT, 'separated')
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, 'analysis')

os.makedirs(ANALYSIS_DIR, exist_ok=True)

def visualize_waveforms(mixture_path, sources_paths, estimated_paths, output_dir=ANALYSIS_DIR):
    # Load audio files
    mixture, sr = librosa.load(mixture_path, sr=None)
    sources = [librosa.load(path, sr=None)[0] for path in sources_paths]
    estimated = [librosa.load(path, sr=None)[0] for path in estimated_paths]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Plot mixture
    plt.subplot(5, 1, 1)
    librosa.display.waveshow(mixture, sr=sr)
    plt.title('Mixture Signal', fontsize=12)
    plt.ylabel('Amplitude')
    
    # Plot true sources
    for i, source in enumerate(sources):
        plt.subplot(5, 1, 2+i)
        librosa.display.waveshow(source, sr=sr)
        plt.title(f'True Source {i+1}', fontsize=12)
        plt.ylabel('Amplitude')
    
    # Plot estimated sources
    for i, est in enumerate(estimated):
        plt.subplot(5, 1, 4+i)
        librosa.display.waveshow(est, sr=sr)
        plt.title(f'Estimated Source {i+1}', fontsize=12)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save figure
    base_name = os.path.basename(mixture_path).replace('mix_', '').replace('.wav', '')
    output_path = os.path.join(output_dir, f'waveforms_{base_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_spectrograms(mixture_path, sources_paths, estimated_paths, output_dir=ANALYSIS_DIR):
    # Load audio files
    mixture, sr = librosa.load(mixture_path, sr=None)
    sources = [librosa.load(path, sr=None)[0] for path in sources_paths]
    estimated = [librosa.load(path, sr=None)[0] for path in estimated_paths]
    
    # Compute spectrograms
    mixture_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(mixture)), ref=np.max)
    
    sources_specs = [
        librosa.amplitude_to_db(np.abs(librosa.stft(s)), ref=np.max)
        for s in sources
    ]
    
    estimated_specs = [
        librosa.amplitude_to_db(np.abs(librosa.stft(e)), ref=np.max)
        for e in estimated
    ]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Plot mixture spectrogram
    plt.subplot(5, 1, 1)
    librosa.display.specshow(mixture_spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mixture Spectrogram', fontsize=12)
    
    # Plot true source spectrograms
    for i, spec in enumerate(sources_specs):
        plt.subplot(5, 1, 2+i)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'True Source {i+1} Spectrogram', fontsize=12)
    
    # Plot estimated source spectrograms
    for i, spec in enumerate(estimated_specs):
        plt.subplot(5, 1, 4+i)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Estimated Source {i+1} Spectrogram', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    base_name = os.path.basename(mixture_path).replace('mix_', '').replace('.wav', '')
    output_path = os.path.join(output_dir, f'spectrograms_{base_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_comparison_audio(mixture_path, sources_paths, estimated_paths, output_dir=ANALYSIS_DIR):
    # Load audio files
    mixture, sr = librosa.load(mixture_path, sr=None)
    sources = [librosa.load(path, sr=None)[0] for path in sources_paths]
    estimated = [librosa.load(path, sr=None)[0] for path in estimated_paths]
    
    # Create comparison files for each source
    output_paths = []
    
    for i in range(len(sources)):
        # Ensure sources have the same length
        min_length = min(len(sources[i]), len(estimated[i]))
        source = sources[i][:min_length]
        estimate = estimated[i][:min_length]
        
        # Create segments of 2 seconds each (alternating between true and estimated)
        segment_length = 2 * sr
        num_segments = min_length // segment_length
        
        if num_segments < 2:
            # If audio is too short, just concatenate true and estimated
            comparison = np.concatenate([
                source, 
                np.zeros(int(0.5 * sr)),  # 0.5s silence between
                estimate
            ])
        else:
            # Create alternating segments
            comparison = np.array([])
            for seg in range(num_segments):
                if seg % 2 == 0:
                    # True source segment
                    segment = source[seg * segment_length:(seg + 1) * segment_length]
                else:
                    # Estimated source segment
                    segment = estimate[seg * segment_length:(seg + 1) * segment_length]
                
                comparison = np.concatenate([comparison, segment])
        
        # Save comparison audio
        base_name = os.path.basename(mixture_path).replace('mix_', '').replace('.wav', '')
        output_path = os.path.join(output_dir, f'compare_source{i+1}_{base_name}.wav')
        sf.write(output_path, comparison, sr)
        output_paths.append(output_path)
    
    return output_paths

def generate_metrics_summary(output_dir=OUTPUT_ROOT):
    # Check for separation metrics
    separation_metrics_path = os.path.join(SEPARATED_DIR, 'test', 'separation_metrics.csv')
    identification_results_path = os.path.join(output_dir, 'identification_results.csv')
    
    if not os.path.exists(separation_metrics_path) or not os.path.exists(identification_results_path):
        print("Required metrics files not found")
        return None
    
    # Load metrics
    separation_metrics = pd.read_csv(separation_metrics_path)
    identification_results = pd.read_csv(identification_results_path)
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'metrics_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPEAKER SEPARATION AND IDENTIFICATION METRICS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Separation metrics
        f.write("1. SOURCE SEPARATION METRICS:\n")
        f.write("-"*50 + "\n")
        for col in separation_metrics.columns:
            f.write(f"  {col:<15}: {separation_metrics[col].values[0]:.4f}\n")
        f.write("\n")
        
        # Interpretation of separation metrics
        f.write("Interpretation:\n")
        f.write("  - SDR (Signal-to-Distortion Ratio): Higher is better. Good: >10, Average: 5-10, Poor: <5\n")
        f.write("  - SIR (Signal-to-Interference Ratio): Higher is better. Measures interference rejection.\n")
        f.write("  - SAR (Signal-to-Artifact Ratio): Higher is better. Measures absence of artifacts.\n")
        f.write("  - PESQ (Perceptual Evaluation of Speech Quality): Range 1-4.5, higher is better.\n\n")
        
        # Identification results
        f.write("2. SPEAKER IDENTIFICATION RESULTS:\n")
        f.write("-"*50 + "\n")
        
        for i, row in identification_results.iterrows():
            model = row['model']
            accuracy = row['accuracy'] * 100
            correct = int(row['correct'])
            total = int(row['total'])
            
            f.write(f"  {model.capitalize()} model:\n")
            f.write(f"    - Accuracy: {accuracy:.2f}%\n")
            f.write(f"    - Correct identifications: {correct}/{total}\n")
        
        if len(identification_results) >= 2:
            pretrained_acc = identification_results.loc[0, 'accuracy']
            finetuned_acc = identification_results.loc[1, 'accuracy']
            improvement = (finetuned_acc - pretrained_acc) * 100
            f.write(f"\n  Improvement from fine-tuning: {improvement:.2f}%\n")
        
        f.write("\n")
        
        # Overall assessment
        f.write("3. OVERALL ASSESSMENT:\n")
        f.write("-"*50 + "\n")
        
        # Evaluate separation quality
        if 'SDR' in separation_metrics.columns:
            sdr = separation_metrics['SDR'].values[0]
            if sdr > 10:
                separation_quality = "Excellent"
            elif sdr > 7:
                separation_quality = "Good"
            elif sdr > 5:
                separation_quality = "Average"
            else:
                separation_quality = "Poor"
            
            f.write(f"  Source separation quality: {separation_quality} (SDR: {sdr:.2f})\n")
        
        # Evaluate identification performance
        if len(identification_results) >= 2:
            finetuned_acc = identification_results.loc[1, 'accuracy'] * 100
            if finetuned_acc > 80:
                identification_quality = "Excellent"
            elif finetuned_acc > 60:
                identification_quality = "Good"
            elif finetuned_acc > 40:
                identification_quality = "Average"
            else:
                identification_quality = "Poor"
            
            f.write(f"  Speaker identification performance: {identification_quality} (Accuracy: {finetuned_acc:.2f}%)\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"Metrics summary saved to {summary_path}")
    return summary_path

def analyze_random_samples(num_samples=5, output_dir=ANALYSIS_DIR):
    # Find test mixtures
    test_metadata_path = os.path.join(MIXTURES_DIR, 'test', 'test_metadata.csv')
    separation_results_path = os.path.join(SEPARATED_DIR, 'test', 'separation_results.csv')
    
    if not os.path.exists(test_metadata_path) or not os.path.exists(separation_results_path):
        print("Required metadata files not found")
        return []
    
    # Load metadata
    separation_results = pd.read_csv(separation_results_path)
    
    # Select random samples
    sample_indices = np.random.choice(
        len(separation_results), 
        size=min(num_samples, len(separation_results)), 
        replace=False
    )
    
    analyzed_samples = []
    
    for idx in sample_indices:
        try:
            sample = separation_results.iloc[idx]
            
            # Get file paths
            mixture_path = sample['mixture']
            source1_path = sample['source1']
            source2_path = sample['source2']
            est1_path = sample['estimated1']
            est2_path = sample['estimated2']
            
            # Skip if any file is missing
            if not all(os.path.exists(p) for p in [mixture_path, source1_path, source2_path, est1_path, est2_path]):
                continue
            
            # Generate visualizations
            waveform_path = visualize_waveforms(
                mixture_path, 
                [source1_path, source2_path], 
                [est1_path, est2_path],
                output_dir
            )
            
            spectrogram_path = visualize_spectrograms(
                mixture_path, 
                [source1_path, source2_path], 
                [est1_path, est2_path],
                output_dir
            )
            
            # Create comparison audio
            comparison_paths = create_comparison_audio(
                mixture_path, 
                [source1_path, source2_path], 
                [est1_path, est2_path],
                output_dir
            )
            
            # Get sample ID
            sample_id = os.path.basename(mixture_path).replace('mix_', '').replace('.wav', '')
            analyzed_samples.append(sample_id)
            
            print(f"Analyzed sample: {sample_id}")
            print(f"  - Waveform visualization: {os.path.basename(waveform_path)}")
            print(f"  - Spectrogram visualization: {os.path.basename(spectrogram_path)}")
            print(f"  - Comparison audio: {', '.join(os.path.basename(p) for p in comparison_paths)}")
            print()
            
        except Exception as e:
            print(f"Error analyzing sample at index {idx}: {e}")
    
    return analyzed_samples

def main():
    parser = argparse.ArgumentParser(description='Visualization utilities for speaker separation')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Command to visualize a specific sample
    vis_parser = subparsers.add_parser('visualize', help='Visualize a specific sample')
    vis_parser.add_argument('--mixture', type=str, required=True, help='Path to mixture audio')
    vis_parser.add_argument('--source1', type=str, required=True, help='Path to first source audio')
    vis_parser.add_argument('--source2', type=str, required=True, help='Path to second source audio')
    vis_parser.add_argument('--est1', type=str, required=True, help='Path to first estimated source')
    vis_parser.add_argument('--est2', type=str, required=True, help='Path to second estimated source')
    
    # Command to analyze random samples
    random_parser = subparsers.add_parser('random', help='Analyze random samples')
    random_parser.add_argument('--count', type=int, default=5, help='Number of random samples to analyze')
    
    # Command to generate metrics summary
    metrics_parser = subparsers.add_parser('metrics', help='Generate metrics summary')
    
    args = parser.parse_args()
    
    if args.command == 'visualize':
        # Visualize specific sample
        visualize_waveforms(
            args.mixture,
            [args.source1, args.source2],
            [args.est1, args.est2]
        )
        
        visualize_spectrograms(
            args.mixture,
            [args.source1, args.source2],
            [args.est1, args.est2]
        )
        
        create_comparison_audio(
            args.mixture,
            [args.source1, args.source2],
            [args.est1, args.est2]
        )
        
        print("Visualizations and comparison audio generated successfully")
        
    elif args.command == 'random':
        # Analyze random samples
        analyzed = analyze_random_samples(num_samples=args.count)
        print(f"Analyzed {len(analyzed)} random samples")
        
    elif args.command == 'metrics':
        # Generate metrics summary
        generate_metrics_summary()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()