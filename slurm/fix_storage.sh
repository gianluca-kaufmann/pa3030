#!/bin/bash
# fix_storage.sh — Move data/ and outputs/ to $SCRATCH, replace with symlinks.
# Run from the Euler login node:  bash ~/master_thesis/slurm/fix_storage.sh
set -euo pipefail

REPO="$HOME/master_thesis"
SCRATCH_BASE="/cluster/scratch/$(whoami)"

cd "$REPO"
echo "=== Euler storage fix ==="
echo "Repo:    $REPO"
echo "Scratch: $SCRATCH_BASE"
echo ""

# ── Pre-flight ──────────────────────────────────────────────────────────
if [ -L "$REPO/data" ] && [ -L "$REPO/outputs" ]; then
    echo "Both data/ and outputs/ are already symlinks. Nothing to do."
    ls -la data outputs
    exit 0
fi

echo "Current home usage:"
du -sh "$REPO/data" "$REPO/outputs" "$REPO/.git" 2>/dev/null || true
echo ""

# ── Step 1: Move data/ to $SCRATCH ─────────────────────────────────────
if [ -d "$REPO/data" ] && [ ! -L "$REPO/data" ]; then
    echo "[1/4] Moving data/ → $SCRATCH_BASE/data ..."
    mkdir -p "$SCRATCH_BASE"
    if [ -d "$SCRATCH_BASE/data" ]; then
        echo "  WARNING: $SCRATCH_BASE/data already exists. Merging with rsync."
        rsync -a --remove-source-files "$REPO/data/" "$SCRATCH_BASE/data/"
        find "$REPO/data" -type d -empty -delete 2>/dev/null || true
        rm -rf "$REPO/data"
    else
        mv "$REPO/data" "$SCRATCH_BASE/data"
    fi
    ln -s "$SCRATCH_BASE/data" "$REPO/data"
    echo "  Done: data → $(readlink "$REPO/data")"
else
    echo "[1/4] data/ already handled (symlink or missing). Skipping."
fi

# ── Step 2: Move outputs/ to $SCRATCH ──────────────────────────────────
if [ -d "$REPO/outputs" ] && [ ! -L "$REPO/outputs" ]; then
    echo "[2/4] Moving outputs/ → $SCRATCH_BASE/outputs ..."
    mkdir -p "$SCRATCH_BASE"
    if [ -d "$SCRATCH_BASE/outputs" ]; then
        echo "  WARNING: $SCRATCH_BASE/outputs already exists. Merging with rsync."
        rsync -a --remove-source-files "$REPO/outputs/" "$SCRATCH_BASE/outputs/"
        find "$REPO/outputs" -type d -empty -delete 2>/dev/null || true
        rm -rf "$REPO/outputs"
    else
        mv "$REPO/outputs" "$SCRATCH_BASE/outputs"
    fi
    ln -s "$SCRATCH_BASE/outputs" "$REPO/outputs"
    echo "  Done: outputs → $(readlink "$REPO/outputs")"
else
    echo "[2/4] outputs/ already handled (symlink or missing). Skipping."
fi

# ── Step 3: Verify symlinks ────────────────────────────────────────────
echo ""
echo "[3/4] Verifying symlinks..."
ls -la "$REPO/data" "$REPO/outputs"
echo ""
echo "Spot-checking paths through symlinks:"
ls "$REPO/outputs/" 2>/dev/null | head -5 && echo "  outputs/ OK" || echo "  outputs/ empty or missing contents"
ls "$REPO/data/"    2>/dev/null | head -5 && echo "  data/ OK"    || echo "  data/ empty or missing contents"

# ── Step 4: Shrink .git ────────────────────────────────────────────────
echo ""
echo "[4/4] Shrinking .git (git gc)..."
cd "$REPO"
git gc --aggressive --prune=now
echo "  .git size after gc: $(du -sh .git | cut -f1)"

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "Home repo size: $(du -sh --exclude=data --exclude=outputs "$REPO" 2>/dev/null | cut -f1 || du -sh "$REPO" | cut -f1)"
echo ""
echo "REMINDER: $SCRATCH files are purged after ~15 days of no access."
echo "Back up critical outputs with:"
echo "  gsutil -m rsync -r $SCRATCH_BASE/outputs gs://protected-areas/outputs"
