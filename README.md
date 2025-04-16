# The Lost One - Facial Kinship Verification System

## Project Overview

"The Lost One" is an advanced computer vision project that aims to identify parent-child relationships through facial feature analysis. The system can determine if a child is biologically related to two adults based solely on facial images, using a combination of geometric facial measurements, color analysis, and deep learning techniques.

The project was developed as my end-of-degree project in Computer Science and demonstrates practical applications of machine learning, computer vision, and biometric analysis.

## Key Features

* **Multi-model Facial Analysis**: Combines traditional geometric measurements with deep learning
* **Triple-subject Kinship Verification**: Analyzes parent-parent-child relationships
* **Fast API Backend**: Scalable RESTful service for face kinship verification
* **Interactive GUI**: Desktop application for testing and visualization
* **Database Integration**: SQLite storage for facial features and embeddings

## Technical Approach

The system uses a multi-faceted approach to kinship verification:

1. **Landmark Detection**: Dlib's 68-point facial landmark detector extracts precise facial geometry
2. **Feature Engineering**: Calculation of facial ratios (e.g., nose-to-face width), angles, and color attributes
3. **Deep Learning Models**: Multiple neural network architectures for kinship verification:
   * Binary parent-child classification
   * Triple subject (father-mother-child) classification
   * ResNet-based feature extraction
4. **Ensemble Method**: Combines predictions from various models for improved accuracy

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
* **Evaluation System**: Comprehensive metrics for model performance

### Technologies Used:

* **Python**: Core programming language
* **OpenCV & Dlib**: Computer vision and facial landmark detection
* **PyTorch & TensorFlow**: Deep learning frameworks
* **FastAPI**: Backend API development
* **SQLite**: Database for facial features storage

## Results and Performance

The system achieved accuracy in identifying parent-child relationships from facial images. The triple-subject classification model demonstrated particularly strong performance in distinguishing true family relationships from random groupings.

Key metrics:
* Binary classification accuracy: ~85%
* Triple-subject classification accuracy: ~80%
* Ensemble model performance: ~88%

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

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python main.py

# Launch the GUI application
python Main_Gui.py
