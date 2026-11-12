import os
import sys
import json
import subprocess
import warnings
import re
from phonemizer import phonemize
from phonemizer.separator import Separator

try:
    import pykakasi
except ImportError:
    pykakasi = None

# On Windows, espeak-ng is often not dynamically added to the PATH for the current session after install.
# We append its default installation path as a fallback.
if sys.platform == 'win32':
    espeak_path = r"C:\Program Files\eSpeak NG"
    if espeak_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = espeak_path + os.pathsep + os.environ.get("PATH", "")
    
    # Phonemizer also needs the exact DLL path on Windows sometimes.
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    dll_path = os.path.join(espeak_path, "libespeak-ng.dll")
    if os.path.exists(dll_path):
        EspeakWrapper.set_library(dll_path)

# We cache at the module level
CACHE_FILE = "phoneme_cache.json"

# Supported language map. Values represent the primary script expected.
SUPPORTED_LANGUAGES = {
    'en-us': 'Latin',
    'es': 'Latin',
    'de': 'Latin',
    'fr-fr': 'Latin',
    'hi': 'Devanagari',
    'ta': 'Tamil',
    'ja': 'Japanese',
    'cmn': 'Chinese', # Mandarin
    'it': 'Latin',
    'ru': 'Cyrillic',
    'pt': 'Latin',
    'ar': 'Arabic',
    'tr': 'Latin',
    'ko': 'Korean',
    'vi': 'Latin',
    'pl': 'Latin',
    'nl': 'Latin',
    'sv': 'Latin',
    'el': 'Greek',
    'id': 'Latin'
}

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# Load cache once on module import
_phoneme_cache = load_cache()

def validate_espeak_languages():
    """Validates if espeak-ng is installed and supports our required languages."""
    try:
        # Check if espeak-ng is available
        result = subprocess.run(['espeak-ng', '--voices'], capture_output=True, text=True, check=True)
        # Not doing strict parsing of output here, but verifying espeak-ng exists.
        # If needed, one could parse 'result.stdout' to ensure 'en-us', 'hi', etc. are present.
    except FileNotFoundError:
        warnings.warn("espeak-ng executable not found in PATH! Phonemizer requires it. Please install espeak-ng.")
    except Exception as e:
        warnings.warn(f"Failed to run espeak-ng --voices: {e}")

# Run validation on import
validate_espeak_languages()

def contains_latin(text: str) -> bool:
    """Checks if the text contains Latin characters."""
    return bool(re.search(r'[A-Za-z]', text))

def preprocess_japanese(text: str) -> str:
    """Uses pykakasi to convert Kanji/Kana into a format reliable for espeak-ng (Hiragana)."""
    if pykakasi is None:
        warnings.warn("pykakasi is not installed. Japanese preprocessing skipped.")
        return text
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    # Join the hiragana representation for better espeak-ng compatibility
    return "".join([item['hira'] for item in result])

def text_to_phonemes(text: str, lang_code: str) -> str:
    """
    Converts text to phonemes using phonemizer (espeak backend) with phone-level separation.
    
    Args:
        text (str): The input text to convert.
        lang_code (str): The target language code (e.g., 'en-us', 'hi', 'ja').
        
    Returns:
        str: Phonemized string with space as phone separator and '|' as word separator.
    """
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{lang_code}' not supported. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
    
    # Check cache
    cache_key = f"{lang_code}::{text}"
    if cache_key in _phoneme_cache:
        return _phoneme_cache[cache_key]

    # Validate input script
    expected_script = SUPPORTED_LANGUAGES[lang_code]
    if expected_script not in ['Latin', 'Japanese']:
        if contains_latin(text):
            warnings.warn(
                f"Text for language '{lang_code}' contains Latin characters, but expects {expected_script} script. "
                "Ensure you are passing native script text, not romanized text!"
            )
            
    # Language specific preprocessing
    processed_text = text
    if lang_code == 'ja':
        processed_text = preprocess_japanese(text)

    # Convert to phonemes
    separator = Separator(phone=' ', word='|')
    try:
        phn = phonemize(
            processed_text,
            language=lang_code,
            backend='espeak',
            separator=separator,
            strip=True,
            preserve_punctuation=True,
            njobs=1
        )
    except Exception as e:
        raise RuntimeError(f"Phonemization failed for {lang_code}. Error: {e}")

    # Update cache
    _phoneme_cache[cache_key] = phn
    save_cache(_phoneme_cache)
    
    return phn
