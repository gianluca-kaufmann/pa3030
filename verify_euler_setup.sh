#!/bin/bash
# Verification script for Euler cluster setup
# Run this interactively on Euler to check your setup before submitting SLURM jobs

echo "========================================================================"
echo "🔍 Euler Cluster Setup Verification for merge_2012_optimized"
echo "========================================================================"
echo ""

# Check SCRATCH variable
echo "1️⃣  Checking SCRATCH environment variable..."
if [ -z "${SCRATCH:-}" ]; then
    echo "  ✗ SCRATCH is not set!"
    echo "    This should be set automatically on Euler compute nodes."
else
    echo "  ✓ SCRATCH = $SCRATCH"
fi
echo ""

# Check data directory
echo "2️⃣  Checking data directories..."
if [ -d "$SCRATCH/data" ]; then
    echo "  ✓ $SCRATCH/data exists"
    
    if [ -d "$SCRATCH/data/ready" ]; then
        echo "  ✓ $SCRATCH/data/ready exists"
        
        # Count datasets
        dataset_count=$(find "$SCRATCH/data/ready" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "    Found $dataset_count dataset directories"
        
        # Check specifically for WDPA
        if [ -d "$SCRATCH/data/ready/WDPA" ]; then
            echo "  ✓ $SCRATCH/data/ready/WDPA exists"
            wdpa_count=$(ls "$SCRATCH/data/ready/WDPA"/*.tif 2>/dev/null | wc -l)
            echo "    Found $wdpa_count WDPA .tif files"
            
            if [ $wdpa_count -gt 0 ]; then
                echo "    Sample files:"
                ls "$SCRATCH/data/ready/WDPA"/*.tif 2>/dev/null | head -3 | sed 's/^/      /'
            else
                echo "  ⚠️  No .tif files found in WDPA directory!"
            fi
        else
            echo "  ✗ $SCRATCH/data/ready/WDPA does NOT exist"
            echo "    Available datasets:"
            ls -1 "$SCRATCH/data/ready" 2>/dev/null | head -10 | sed 's/^/      /'
        fi
    else
        echo "  ✗ $SCRATCH/data/ready does NOT exist"
    fi
else
    echo "  ✗ $SCRATCH/data does NOT exist"
    echo "    You need to copy your data to $SCRATCH/data"
fi
echo ""

# Check W&B authentication
echo "3️⃣  Checking Weights & Biases authentication..."
if [ -f "$HOME/.netrc" ]; then
    echo "  ✓ Found $HOME/.netrc (credentials from 'wandb login')"
    
    # Check if wandb API is in .netrc
    if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        echo "  ✓ W&B credentials found in .netrc"
    else
        echo "  ⚠️  .netrc exists but may not contain W&B credentials"
    fi
else
    echo "  ✗ No $HOME/.netrc found"
    echo "    Run 'wandb login' to create it"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  ✓ WANDB_API_KEY is set in environment"
else
    echo "  ℹ️  WANDB_API_KEY not set (will use .netrc if available)"
fi
echo ""

# Check virtual environment
echo "4️⃣  Checking virtual environment..."
VENV_DIR="$HOME/venv/master-thesis"
if [ -d "$VENV_DIR" ]; then
    echo "  ✓ Virtual environment exists at $VENV_DIR"
    
    if [ -f "$VENV_DIR/bin/python" ]; then
        echo "  ✓ Python executable found"
        
        # Try activating and checking packages
        if source "$VENV_DIR/bin/activate" 2>/dev/null; then
            echo "  ✓ Can activate virtual environment"
            
            # Check for required packages
            echo "    Checking required packages..."
            for pkg in wandb rasterio numpy pandas pyarrow; do
                if python -c "import $pkg" 2>/dev/null; then
                    version=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
                    echo "      ✓ $pkg ($version)"
                else
                    echo "      ✗ $pkg not installed"
                fi
            done
            
            deactivate 2>/dev/null
        fi
    fi
else
    echo "  ✗ Virtual environment not found at $VENV_DIR"
fi
echo ""

# Check project directory
echo "5️⃣  Checking project directory..."
PROJECT_DIR="$HOME/master_thesis"
if [ -d "$PROJECT_DIR" ]; then
    echo "  ✓ Project directory exists at $PROJECT_DIR"
    
    if [ -f "$PROJECT_DIR/scripts/merging/merge_2012_optimized" ]; then
        echo "  ✓ merge_2012_optimized script found"
    else
        echo "  ✗ merge_2012_optimized script NOT found"
    fi
else
    echo "  ✗ Project directory not found at $PROJECT_DIR"
fi
echo ""

# Summary
echo "========================================================================"
echo "📋 Summary"
echo "========================================================================"

# Determine readiness
ready=true

if [ -z "${SCRATCH:-}" ]; then
    echo "❌ SCRATCH variable not set"
    ready=false
fi

if [ ! -d "$SCRATCH/data/ready/WDPA" ] || [ $(ls "$SCRATCH/data/ready/WDPA"/*.tif 2>/dev/null | wc -l) -eq 0 ]; then
    echo "❌ WDPA data not found in $SCRATCH/data/ready/WDPA"
    ready=false
fi

if [ ! -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "❌ W&B authentication not configured"
    ready=false
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found"
    ready=false
fi

if $ready; then
    echo "✅ All checks passed! You're ready to submit the SLURM job."
    echo ""
    echo "To submit:"
    echo "  cd $PROJECT_DIR"
    echo "  sbatch merge_2012.slurm"
else
    echo ""
    echo "⚠️  Some issues found. Please fix them before submitting the job."
    echo ""
    echo "Common fixes:"
    echo "  • Copy data: rsync -av ~/Desktop/Master\\'s\\ Thesis/code/data/ \$SCRATCH/data/"
    echo "  • W&B login: wandb login"
    echo "  • Create venv: see create_env.sh"
fi
echo ""

