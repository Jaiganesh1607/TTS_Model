import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import warnings

# Global model variables
_model = None
_processor = None
_device = None
TARGET_SAMPLE_RATE = 16000

def _init_model():
    global _model, _processor, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Feature Extractor] Loading XLS-R 300m on {_device}...")
    
    # We suppress some huggingface warnings about uninitialized weights which are normal for feature extraction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        _model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
    
    if _device.type == "cuda":
        print("[Feature Extractor] Casting model to FP16 to save VRAM...")
        _model = _model.half()  # Convert to FP16 only if using CUDA
        
    _model = _model.to(_device)
    _model.eval()
    
    # Freeze all parameters explicitly
    for param in _model.parameters():
        param.requires_grad = False
        
    print("[Feature Extractor] Model loaded and frozen successfully.\n")

def extract_features(audio_path: str) -> torch.Tensor:
    """
    Loads an audio file, normalizes/resamples, and extracts XLS-R features.
    
    Args:
        audio_path (str): Path to the audio file.
        
    Returns:
        torch.Tensor: The extracted hidden states.
    """
    _init_model()
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # If stereo, convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample to 16000 Hz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze(0).numpy()
    
    # The processor handles zero-mean, unit-variance normalization which Wav2Vec2 requires
    inputs = _processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(_device)
    
    if _device.type == "cuda":
        input_values = input_values.half()
        
    # Extract features
    with torch.no_grad():
        outputs = _model(input_values)
        
    # Extract the hidden states and remove the batch dimension
    hidden_states = outputs.last_hidden_state.squeeze(0)
    
    # Return to CPU to prevent VRAM memory leaks during huge batch processing
    return hidden_states.cpu()
