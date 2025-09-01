
CloudFactory Object Detection Pipeline: 
Installation & Setup (GitHub) 
Dependencies Installation
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
while using docker I had to take detectron2==0.6 out however in the docker I have wrote to pip install it should work however if it doesn’t need to pip install detectron2==0.6.

now pull the github repository file : https://github.com/Samyak44/ImageRecognition_pipeline

Important: Use numpy<2.0.0 to avoid compatibility issues with PyTorch.
Basic Execution

# Run with default settings
python app.py

# Process images with custom confidence threshold
python app.py --confidence_threshold 0.5

# Adjust NMS threshold for detection overlap
python app.py --nms_threshold 0.3

# Enable debug logging
python app.py --debug
To run according we to pass the parameters: this will create the output. 
python app.py --images_dir ./Image --model_path ./fish_detector/model.pt --class_mapping_path ./fish_detector/class_mapping.json --output_dir ./output
When dockerized: 
docker pull mavrick444/fish-imagedetection:latest
Create the following folders on your local machine:
fish-detection-demo/
├── images/              # Put your fish images here (.jpg, .png)
├── models/              # Model files (see below)
│   ├── model.pth
│   └── class_mapping.json
└── output/              # Results will appear here

Running the pipeline: 
docker run -v $(pwd)/images:/app/images \ -v $(pwd)/models:/app/models \ -v $(pwd)/output:/app/output \ mavrick444/fish-imagedetection:latest \ python app.py \ --images_dir /app/images \ --model_path /app/models/model.pth \ --class_mapping_path /app/models/class_mapping.json \ --output_dir /app/output


When you run the image : 
docker run image-detection python app.py \ --images_dir /path/to/images \ --model_path /path/to/model \ --class_mapping_path /path/to/class_mapping \ --output_dir /path/to/output
 
Fish-imagedDectector

This project is a Dockerized pipeline for detecting fish in images using PyTorch.  
The pipeline reads images from the `Image/` folder, runs inference using a pre-trained model in `fish_detector/`, and outputs results to the `output/` folder.
##  Features

- Runs completely inside a Docker container.
- No installation of Python dependencies on the host required.
- Prepares outputs automatically in the `output/` folder.


##  Known Dependency Warning

When running the pipeline, you may see a warning like:

Failed to initialize NumPy: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x...


This happens because:

- PyTorch was compiled with NumPy 1.x.
- Docker image currently has NumPy 2.2.6 (required for OpenCV 4.12+).

This is a warning only  the pipeline has been tested and runs successfully. Outputs are correctly generated.

##  Running the Docker Container

### Build Docker image (if needed):


docker build -t fish-imagedetector .

.github will tigger github action one pushed in main branch
