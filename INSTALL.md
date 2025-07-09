# Installation Guide

This RAG chatbot application requires both Python dependencies and system packages for full functionality.

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies (for OCR support)

For OCR functionality with scanned PDFs, you need to install system packages:

```bash
./install_ocr.sh
```

This script will install:
- **Tesseract OCR**: For optical character recognition
- **Poppler utilities**: For PDF to image conversion
- **Korean language pack**: For Korean text recognition

### Alternative: Use Docker (Coming Soon)

A Docker image with all dependencies pre-installed will be available soon.

## Platform-Specific Notes

### macOS
- Requires Homebrew for system package installation
- All dependencies can be installed via the provided scripts

### Linux (Ubuntu/Debian)
- Uses apt-get for system packages
- May require sudo permissions

### Linux (RedHat/CentOS)
- Uses yum for system packages
- May require sudo permissions

### Windows
- Not officially supported yet
- Consider using WSL2 (Windows Subsystem for Linux)

## Dependency Details

### Python Packages (via pip)
- **Core**: streamlit, langchain ecosystem
- **Vector DB**: faiss-cpu, chromadb
- **ML/AI**: sentence-transformers, torch, transformers
- **PDF Processing**: pypdf, PyPDF2, pdf2image[jpeg]
- **OCR**: pytesseract (Python wrapper for Tesseract)
- **API Clients**: openai, anthropic, google-generativeai

### System Packages (via package manager)
- **tesseract-ocr**: OCR engine
- **tesseract-ocr-kor**: Korean language support
- **poppler-utils**: PDF rendering (required by pdf2image)

## Verification

After installation, verify everything is working:

```bash
# Check Python dependencies
python -c "import streamlit, langchain, pytesseract; print('Python dependencies OK')"

# Check system dependencies
tesseract --version
pdftoppm -v
```

## Troubleshooting

### OCR not working
- Ensure Tesseract is installed: `which tesseract`
- Check language packs: `tesseract --list-langs`

### PDF to image conversion failing
- Ensure poppler is installed: `which pdftoppm`
- On macOS: `brew install poppler`
- On Linux: `sudo apt-get install poppler-utils`

### Import errors
- Ensure all packages installed: `pip install -r requirements.txt`
- Check Python version: Requires Python 3.8+