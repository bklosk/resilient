#!/bin/bash

# Define the path to the virtual environment
VENV_PATH="./venv"

# Check if python3.12 command is available
if ! command -v python3.12 &> /dev/null
then
    echo "python3.12 could not be found."
    echo "Please install Python 3.12 or ensure it's in your PATH."
    exit 1
fi

# Check if the virtual environment directory exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH."
    echo "Attempting to create it using python3.12..."
    python3.12 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment with Python 3.12."
        echo "Please check your Python 3.12 installation and try creating the venv manually:"
        echo "python3.12 -m venv $VENV_PATH"
        exit 1
    fi
    echo "Virtual environment created successfully with Python 3.12."
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

echo "Virtual environment activated."

# Define the path to your Python script
SCRIPT_PATH="./process_point_cloud.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Python script not found at $SCRIPT_PATH"
    deactivate
    exit 1
fi

# Run the colorize command
echo "Running colorization..."
python "$SCRIPT_PATH" colorize \
    --input_pc point_cloud.laz \
    --input_sat boulder_flyover/images/n2w235.tif \
    --output_pc colorized_point_cloud.laz \
    --x_offset -2 \
    --y_offset 0

# Check if colorization was successful (basic check: output file exists)
if [ ! -f "colorized_point_cloud.laz" ]; then
    echo "Colorization failed or output file 'colorized_point_cloud.laz' not found."
    deactivate
    exit 1
fi

# Run the visualize command
echo "Running visualization..."
python "$SCRIPT_PATH" visualize --file colorized_point_cloud.laz

# Deactivate the virtual environment (optional, as it deactivates on script exit)
# deactivate
echo "Script execution finished."

