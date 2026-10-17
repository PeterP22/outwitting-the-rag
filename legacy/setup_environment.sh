#!/bin/bash
# Setup script for Outwitting the Devil RAG project
# This script creates a virtual environment and installs dependencies

echo "Setting up RAG project environment..."
echo "===================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Display Python version
echo "Python version:"
python3 --version
echo ""

# Create virtual environment
ENV_NAME="venv"
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment already exists. Removing old environment..."
    rm -rf "$ENV_NAME"
fi

echo "Creating virtual environment..."
python3 -m venv "$ENV_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "    deactivate"
echo ""
echo "Next step: Run the PDF extraction test:"
echo "    python extract_pdf_test.py"