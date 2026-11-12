import os
import glob
import torch
from tqdm import tqdm
from feature_extractor import extract_features

DATA_DIR = "data"
FEATURES_DIR = "features"

def main():
    # Find all .wav files in the data directory and subdirectories
    search_pattern = os.path.join(DATA_DIR, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {DATA_DIR}/. Please place your audio files there.")
        return
        
    print(f"Found {len(wav_files)} audio files to process.")
    
    # Process files one by one to save memory
    for wav_path in tqdm(wav_files, desc="Precomputing Features"):
        # Map data/lang/file.wav to features/lang/file.pt
        rel_path = os.path.relpath(wav_path, DATA_DIR)
        feat_path = os.path.join(FEATURES_DIR, rel_path).replace(".wav", ".pt")
        
        # Check if already cached (resumable)
        if os.path.exists(feat_path):
            continue
            
        # Ensure target directory exists
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        
        try:
            # Extract features
            features = extract_features(wav_path)
            
            # Save tensor
            torch.save(features, feat_path)
            
            # Explicitly free memory (garbage collector usually handles this, but good practice for VRAM)
            del features
        except Exception as e:
            import traceback
            print(f"\nFailed to process {wav_path}:")
            traceback.print_exc()

if __name__ == "__main__":
    main()
