#!/bin/bash
# Quick deployment script for Call Center Forecast Dashboard

echo "============================================"
echo "Call Center Forecast - Deployment Helper"
echo "============================================"
echo ""

# Check if data exists
if [ ! -f "data/combined_data.csv" ]; then
    echo "📊 Generating synthetic data..."
    python generate_data.py
    echo ""
fi

# Check Docker availability
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected. Options:"
    echo ""
    echo "Build Docker image:"
    echo "  docker build -t call-center-forecast ."
    echo ""
    echo "Run locally:"
    echo "  docker run -p 8501:8501 call-center-forecast"
    echo ""
fi

echo "📦 Deployment Options:"
echo ""
echo "1. Streamlit Cloud (Easiest):"
echo "   - Push to GitHub"
echo "   - Go to share.streamlit.io"
echo "   - Connect repository and deploy"
echo ""
echo "2. Docker:"
echo "   - Build: docker build -t call-center-forecast ."
echo "   - Run: docker run -p 8501:8501 call-center-forecast"
echo ""
echo "3. Local Development:"
echo "   - streamlit run dashboard.py"
echo ""
echo "For detailed instructions, see DEPLOYMENT.md"
echo ""

