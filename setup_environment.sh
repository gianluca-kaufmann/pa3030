#!/bin/bash

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_NAME=$(basename "$SCRIPT_DIR")
ENV_NAME="${PROJECT_NAME}_env"
ENV_YML="${SCRIPT_DIR}/environment.yml"

# Sibling directory settings
SEALS_DIR_NAME="seals_dev"
SEALS_REPO_URL="https://github.com/jandrewjohnson/seals_dev.git"
# Path to ../seals_dev
SEALS_FULL_PATH="$(dirname "$SCRIPT_DIR")/${SEALS_DIR_NAME}"

# ANSI Colors for nicer output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n=== SEALS Bonn environment setup (Linux/macOS) ==="

# --- 1. Check Tooling ---

# Check for Mamba / Micromamba
if command -v mamba &> /dev/null; then
    MAMBA="mamba"
elif command -v micromamba &> /dev/null; then
    MAMBA="micromamba"
else
    echo -e "${RED}ERROR: mamba (or micromamba) not found on PATH.${NC}"
    echo "Please install Mambaforge or Micromamba and re-run."
    exit 1
fi

# Check for Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git not found on PATH.${NC}"
    exit 1
fi

echo "Script directory: $SCRIPT_DIR"
echo "Sibling Project:  $SEALS_FULL_PATH"
echo "Conda Env Name:   $ENV_NAME"

# --- 2. Clone Sibling Repo ---
echo -e "\nStep 1/3: Checking local dependencies..."

if [ -d "$SEALS_FULL_PATH" ]; then
    echo -e "  ${GREEN}[OK]${NC} '$SEALS_DIR_NAME' already exists locally."
else
    echo -e "  ${YELLOW}[UPDATE]${NC} '$SEALS_DIR_NAME' not found. Cloning from GitHub..."
    git clone "$SEALS_REPO_URL" "$SEALS_FULL_PATH"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to clone seals_dev.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}[OK]${NC} Clone successful."
fi

# --- 3. Conda Environment ---
echo -e "\nStep 2/3: Creating or updating conda environment..."

if [ ! -f "$ENV_YML" ]; then
    echo -e "${RED}ERROR: environment.yml not found at: $ENV_YML${NC}"
    exit 1
fi

# Switch to script dir so relative paths in yml work
cd "$SCRIPT_DIR" || exit

# Check if env exists
if "$MAMBA" env list | grep -q "$ENV_NAME"; then
    echo "  Env exists; updating (with --prune)..."
    "$MAMBA" env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
    echo "  Env not found; creating..."
    "$MAMBA" env create -n "$ENV_NAME" -f "$ENV_YML"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Conda environment creation/update failed.${NC}"
    exit 1
fi

# --- 4. VS Code Helper ---
echo -e "\nStep 3/3: Configuring VS Code helper..."

# Create .env file for PYTHONPATH
echo "PYTHONPATH=../$SEALS_DIR_NAME" > .env

echo -e "  ${GREEN}[OK]${NC} Created .env file pointing to $SEALS_DIR_NAME"

# --- Done ---
echo -e "\n=========================================================="
echo -e "${GREEN}✅ SETUP COMPLETE${NC}"
echo -e "=========================================================="
echo ""
echo "1. To activate the environment:"
echo "   mamba activate $ENV_NAME" # or conda activate
echo ""
echo "2. VS Code / Pylance Setup:"
echo "   - Open Command Palette (Cmd+Shift+P)"
echo "   - Type: 'Python: Select Interpreter'"
echo "   - Select: '$ENV_NAME'"
echo "   - Pylance will now automatically find 'seals' in ../$SEALS_DIR_NAME"
echo ""
echo -e "=========================================================="