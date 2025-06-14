#!/bin/bash

# Script to clone and set up the adaptation repository in this codespace
# Usage: ./setup-adaptation-repo.sh

echo "🔧 Setting up bklosk/adaptation repository in this codespace..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: This script should be run from the photogrammetry repository root"
    exit 1
fi

# Navigate to the parent workspaces directory
cd /workspaces

# Clone the adaptation repository if it doesn't exist
if [ ! -d "adaptation" ]; then
    echo "📦 Cloning bklosk/adaptation repository..."
    git clone https://github.com/bklosk/adaptation.git
    if [ $? -eq 0 ]; then
        echo "✅ Successfully cloned bklosk/adaptation"
    else
        echo "❌ Failed to clone repository. Check your permissions."
        exit 1
    fi
else
    echo "📁 Repository already exists at /workspaces/adaptation"
fi

# Navigate to the adaptation directory
cd adaptation

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing adaptation repository dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "ℹ️  No requirements.txt found in adaptation repository"
fi

# Set up any additional configuration if needed
echo "🔧 Setting up development environment for adaptation..."

# Make the script executable
chmod +x setup-adaptation-repo.sh 2>/dev/null || true

echo ""
echo "🎉 Setup complete! You can now work with both repositories:"
echo "   📁 Photogrammetry: /workspaces/photogrammetry"
echo "   📁 Adaptation:     /workspaces/adaptation"
echo ""
echo "💡 Tip: Use 'cd /workspaces/adaptation' to switch to the adaptation repository"
echo "💡 Tip: Both repositories are now accessible in VS Code's file explorer"
