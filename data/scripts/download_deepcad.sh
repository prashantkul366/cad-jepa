#!/usr/bin/env bash
# Run from cad-jepa/ root:  bash data/scripts/download_deepcad.sh
set -e
mkdir -p data
echo "Downloading DeepCAD dataset (~2.5 GB)..."
wget -c "http://www.cs.columbia.edu/cg/deepcad/data.tar" -O data/data.tar
echo "Extracting..."
tar -xf data/data.tar -C data/
rm data/data.tar
echo "Done. data/cad_json/ and data/cad_vec/ are ready."
