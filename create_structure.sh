#!/bin/bash
echo "Creating folder structure..."

# Create the main directory structure
mkdir data models notebooks output reports runs src

# Create subdirectories
mkdir data/external data/processed data/raw
mkdir src/data src/features src/models src/visualization

# Create placeholder Python scripts
# touch src/data/01.create_data.py
# touch src/features/01_calculate_mean_std.py
# touch src/models/01_train.py
# touch src/models/helper.py

# Create .gitignore file
cat <<EOF > .gitignore
data/
output/
models/
notebook/
# runs/
src/models/__pycache__/
src/__pycache__/
build/

# Include the data folder specifically located at src/data
!src/data/
!src/models/
EOF

echo "Folder structure created successfully."

