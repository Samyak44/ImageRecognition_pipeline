# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and data folders
COPY app.py .
COPY Image/ ./Image/
COPY fish_detector/ ./fish_detector/

# Create output folder
RUN mkdir -p ./output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run your pipeline
# Fixed: Use app.py instead of pipeline.py, and correct folder names
CMD ["python", "app.py", \
     "--Image_dir", "./Image", \
     "--model_path", "./fish_detector/model.pt", \
     "--class_mapping_path", "./fish_detector/class_mapping.json", \
     "--output_dir", "./output"]