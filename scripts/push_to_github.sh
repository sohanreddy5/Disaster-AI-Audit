#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Push Disaster AI Audit Project to GitHub
# ==============================================================================

set -e

# Target directory on Desktop
PROJECT_DIR="$HOME/Desktop/disaster-ai-audit"
if [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="$HOME/.gemini/antigravity/scratch/disaster-ai-audit"
fi

cd "$PROJECT_DIR"

echo "========================================================================"
echo "   PUSH PROJECT TO GITHUB"
echo "   Local Repository: $PROJECT_DIR"
echo "   Current Branch:   $(git branch --show-current)"
echo "========================================================================"
echo ""

# Check existing remote
EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

if [ -n "$EXISTING_REMOTE" ]; then
    echo "Current remote origin: $EXISTING_REMOTE"
    read -p "Do you want to push to this remote? (y/n): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo "Pushing to $EXISTING_REMOTE..."
        git push -u origin main
        echo ""
        echo "[SUCCESS] Project successfully pushed to GitHub!"
        exit 0
    fi
fi

echo "Please enter your GitHub repository URL."
echo "Examples:"
echo "  HTTPS: https://github.com/<username>/disaster-ai-audit.git"
echo "  SSH:   git@github.com:<username>/disaster-ai-audit.git"
echo ""
read -p "Repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "[!] Error: No repository URL provided."
    exit 1
fi

if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi

echo ""
echo "Pushing 'main' branch to origin ($REPO_URL)..."
git push -u origin main

echo ""
echo "========================================================================"
echo "[SUCCESS] Project successfully pushed to your GitHub repository!"
echo "URL: $REPO_URL"
echo "========================================================================"
