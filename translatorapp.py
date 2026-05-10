from deep_translator import GoogleTranslator

# Simple usage
translated = GoogleTranslator(source='auto', target='b').translate("Hello World")
print(translated)
