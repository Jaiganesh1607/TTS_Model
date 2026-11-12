import os
import torch
import torchaudio
import math

DATA_DIR = "data"

def create_dummy_audio(filename, duration=1.0, freq=440.0, sample_rate=22050):
    """Generates a simple sine wave and saves it as a .wav file."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * math.pi * freq * t).unsqueeze(0)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchaudio.save(filename, waveform, sample_rate)

def main():
    print("Setting up dummy test environment...")
    test_files = [
        os.path.join(DATA_DIR, "en", "dummy1.wav"),
        os.path.join(DATA_DIR, "es", "dummy2.wav"),
        os.path.join(DATA_DIR, "hi", "dummy3.wav"),
    ]
    
    for f in test_files:
        create_dummy_audio(f)
        print(f"Created {f}")
        
    print("\nRunning precompute_features.py...")
    
    # Run the main script function
    import precompute_features
    precompute_features.main()
    
    print("\nVerifying outputs...")
    for f in test_files:
        feat_path = f.replace("data", "features").replace(".wav", ".pt")
        if os.path.exists(feat_path):
            # Load weights_only=True for security (standard in newer PyTorch)
            tensor = torch.load(feat_path, weights_only=True)
            print(f"Success! {feat_path} generated with shape {tensor.shape}")
        else:
            print(f"Error! {feat_path} was not generated.")
            
if __name__ == "__main__":
    main()
