#!/bin/bash

echo "🔧 OCR 도구 설치 스크립트"
echo "이 스크립트는 Tesseract OCR과 한국어 언어팩을 설치합니다."

# OS 확인
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "🍎 macOS 환경 감지됨"
    
    # Homebrew 확인
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew가 설치되어 있지 않습니다."
        echo "먼저 Homebrew를 설치해주세요: https://brew.sh"
        exit 1
    fi
    
    # Tesseract 설치
    echo "📦 Tesseract 설치 중..."
    brew install tesseract
    
    # 한국어 언어팩 설치
    echo "🇰🇷 한국어 언어팩 설치 중..."
    brew install tesseract-lang
    
    # poppler 설치 (pdf2image 의존성)
    echo "📄 Poppler 설치 중..."
    brew install poppler
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Linux 환경 감지됨"
    
    # 패키지 매니저 확인
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "📦 Tesseract 설치 중..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
        
        echo "📄 Poppler 설치 중..."
        sudo apt-get install -y poppler-utils
        
    elif command -v yum &> /dev/null; then
        # RedHat/CentOS
        echo "📦 Tesseract 설치 중..."
        sudo yum install -y tesseract tesseract-langpack-kor tesseract-langpack-eng
        
        echo "📄 Poppler 설치 중..."
        sudo yum install -y poppler-utils
    else
        echo "❌ 지원되지 않는 Linux 배포판입니다."
        exit 1
    fi
else
    echo "❌ 지원되지 않는 운영체제입니다."
    exit 1
fi

# 설치 확인
echo ""
echo "✅ 설치 확인 중..."

if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract 설치됨: $(tesseract --version | head -n 1)"
    
    # 한국어 언어팩 확인
    if tesseract --list-langs 2>&1 | grep -q "kor"; then
        echo "✓ 한국어 언어팩 설치됨"
    else
        echo "❌ 한국어 언어팩이 설치되지 않았습니다."
    fi
else
    echo "❌ Tesseract 설치 실패"
fi

if command -v pdftoppm &> /dev/null; then
    echo "✓ Poppler 설치됨"
else
    echo "❌ Poppler 설치 실패"
fi

echo ""
echo "🎉 OCR 도구 설치 완료!"
echo "이제 스캔된 PDF 파일에서도 텍스트를 추출할 수 있습니다."