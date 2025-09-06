import ssl
import nltk

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Download punkt tokenizer
nltk.download('punkt')
