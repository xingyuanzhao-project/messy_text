import pdfplumber
import sys

# Set default encoding to utf-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

pdf_path = 'Coding manual english.pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                print(text)
except Exception as e:
    print(f"Error processing {pdf_path}: {e}", file=sys.stderr)
