#!/bin/bash
set -eu

# Configuration
REPO="amgadtewfik/atf"
VERSION=${1:-v0.11.0}
ASSET=${2:-"ATF Chat-arm64.dmg"}

echo "Targeting release $VERSION in $REPO"
echo "Asset: $ASSET"

if [ ! -f "$ASSET" ]; then
    echo "Error: Asset file $ASSET not found."
    exit 1
fi

# Check if release exists
if gh release view "$VERSION" --repo "$REPO" >/dev/null 2>&1; then
    echo "Release $VERSION already exists. Uploading asset..."
    gh release upload "$VERSION" "$ASSET" --repo "$REPO"
else
    echo "Release $VERSION does not exist. Creating release and uploading asset..."
    # Use current commit as target, fallback to 'main'
    TARGET=$(git rev-parse HEAD 2>/dev/null || echo "main")
    gh release create "$VERSION" "$ASSET" \
        --repo "$REPO" \
        --target "$TARGET" \
        --title "ATF Chat $VERSION" \
        --generate-notes
fi

echo "Successfully uploaded $ASSET to $VERSION"
