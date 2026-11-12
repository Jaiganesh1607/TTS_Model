import sys
from phoneme_converter import text_to_phonemes, SUPPORTED_LANGUAGES

# Sample sentences for each language
TEST_DATA = {
    'en-us': "Hello world, this is a test.",
    'es': "Hola mundo, esto es una prueba.",
    'de': "Hallo Welt, das ist ein Test.",
    'fr-fr': "Bonjour le monde, c'est un test.",
    'hi': "नमस्ते दुनिया, यह एक परीक्षण है।",
    'ta': "வணக்கம் உலகம், இது ஒரு சோதனை.",
    'ja': "こんにちは世界、これはテストです。",
    'cmn': "你好，世界，这是一个测试。",
    'it': "Ciao mondo, questo è un test.",
    'ru': "Привет мир, это тест.",
    'pt': "Olá mundo, este é um teste.",
    'ar': "مرحبا بالعالم، هذا اختبار.",
    'tr': "Merhaba dünya, bu bir testtir.",
    'ko': "안녕 세상, 이것은 테스트입니다.",
    'vi': "Chào thế giới, đây là một bài kiểm tra.",
    'pl': "Witaj świecie, to jest test.",
    'nl': "Hallo wereld, dit is een test.",
    'sv': "Hej världen, det här är ett test.",
    'el': "Γεια σου κόσμε, αυτό είναι ένα τεστ.",
    'id': "Halo dunia, ini adalah tes.",
    'en-us': "hello my name is Jaiganesh"
}

def main():
    # Force UTF-8 output on Windows consoles to prevent UnicodeEncodeError
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print(f"{'Language':<8} | {'Input Text':<40} | {'Phonemes'}")
    print("-" * 100)
    
    errors = 0
    for lang, text in TEST_DATA.items():
        if lang not in SUPPORTED_LANGUAGES:
            continue
            
        try:
            phonemes = text_to_phonemes(text, lang)
            print(f"{lang:<8} | {text:<40} | {phonemes}")
        except Exception as e:
            print(f"{lang:<8} | {text:<40} | ERROR: {e}")
            errors += 1
            
    if errors > 0:
        print(f"\nCompleted with {errors} errors.")
        sys.exit(1)
    else:
        print("\nAll languages phonemized successfully.")

if __name__ == "__main__":
    main()
