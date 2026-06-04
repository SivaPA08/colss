#!/usr/bin/env bash
set -e

echo "=== Extracting project ==="
unzip -o colss.zip

echo "=== Installing build dependencies ==="
apt-get update
apt-get install -y build-essential cmake

echo "=== Installing Python dependencies ==="
python -m pip install -U pip
python -m pip install scikit-build-core pybind11 numpy build wheel

echo "=== Building and installing colss ==="
cd colss
python -m pip install .

echo "=== Leaving source directory to avoid shadowing ==="
cd /content

echo "=== Testing installation ==="
python -c "
import colss
import numpy as np

print('Imported from:', colss.**file**)

a = np.arange(5.0)
b = np.arange(5.0)

try:
print(colss.query('a+b', a=a, b=b))
except Exception as e:
print('Import succeeded. Query test:', e)
"
