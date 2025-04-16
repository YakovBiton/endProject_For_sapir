# The Lost One - Facial Kinship Verification System

## Project Overview

"The Lost One" is an advanced computer vision project that aims to identify parent-child relationships through facial feature analysis. The system can determine if a child is biologically related to two adults based solely on facial images, using a combination of geometric facial measurements, color analysis, and deep learning techniques.

The project was developed as my end-of-degree project in Computer Science and demonstrates practical applications of machine learning, computer vision, and biometric analysis.

## Key Features

* **Multi-model Facial Analysis**: Combines traditional geometric measurements with deep learning
* **Triple-subject Kinship Verification**: Analyzes parent-parent-child relationships
* **Fast API Backend**: Scalable RESTful service for face kinship verification
* **Interactive GUI**: Desktop application for testing and visualization
* **Web Application**: React-based front-end with authentication for uploading and processing images
* **Database Integration**: SQLite storage for facial features and embeddings

## Technical Approach

The system uses a multi-faceted approach to kinship verification:

1. **Landmark Detection**: Dlib's 68-point facial landmark detector extracts precise facial geometry
2. **Feature Engineering**: Calculation of facial ratios (e.g., nose-to-face width), angles, and color attributes
3. **Deep Learning Models**: Multiple neural network architectures for kinship verification:
   * Binary parent-child classification
   * Triple subject (father-mother-child) classification
   * ResNet-based feature extraction
4. **Ensemble Method**: Combines predictions from various models through a point-based scoring system

## Dataset

The project utilizes the TSKinFace dataset (Tri-Subject Kinship Face), which contains facial images of family trios including:
* Father-Mother-Daughter (FMD)
* Father-Mother-Son (FMS) 
* Father-Mother-Son-Daughter (FMSD)

## Implementation Details

### Core Components:

* **Feature Extraction**: Geometric measurements, color analysis, and neural embeddings
* **Classification Models**: Both PyTorch and TensorFlow/Keras implementations
* **Visualization Tools**: Interactive displays of facial landmarks and features
* **Evaluation System**: Point-based scoring from multiple classifier outputs
* **Web Interface**: React application for uploading parent images and finding potential matches

### Technologies Used:

* **Python**: Core programming language
* **OpenCV & Dlib**: Computer vision and facial landmark detection
* **PyTorch & TensorFlow**: Deep learning frameworks
* **FastAPI**: Backend API development
* **SQLite**: Database for facial features storage
* **React**: Frontend web application
* **Firebase**: Authentication for web application

## Results and Performance

The system uses a point-based approach where each classifier (PyTorch, Keras, ResNet50, etc.) contributes points to potential parent-child matches. Based on the accumulated points, matches are classified as either "potential matches" or "strong matches" if they exceed a certain threshold.

In testing with real data:
* **Potential Matches**: 26% of correct children identified
* **Strong Matches**: 35% of correct children identified
* **Overall Success Rate**: 61% of parent-child relationships correctly identified

## Web Application

The system includes a React-based web application where users can:
* Create an account and log in securely through Firebase authentication
* Upload facial images of two parents (father and mother)
* Process the images through the facial kinship verification system
* View potential children matches ranked by likelihood of biological relation

## Future Directions

* Integration with larger facial databases
* Implementation of more sophisticated deep learning architectures
* Development of privacy-preserving facial analysis techniques
* Mobile application development

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/the-lost-one.git
cd the-lost-one

# Install backend dependencies
pip install -r requirements.txt

# Run the FastAPI server
python main.py

# Launch the GUI application
python Main_Gui.py

# For the React web application
cd webapp
npm install
npm start
