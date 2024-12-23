# Alzheimer's Disease Detection Project

This project utilizes deep learning techniques, specifically **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks, to classify Alzheimer's disease stages based on MRI images. The hybrid architecture extracts both spatial and sequential features, enabling accurate classification.

---

## Features
- **Stage Classification**: Predicts one of the following stages:
  - Mild
  - Moderate
  - Non-Demented
  - Very Mild
- **Model Architecture**: Combines CNN for spatial feature extraction and LSTM for sequential pattern analysis.
- **Frameworks and Libraries Used**: TensorFlow for model development, OpenCV for image preprocessing, and NumPy for efficient data handling.

---

## Implementation Details

### 1. Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**:
  - TensorFlow: For building and training the CNN-LSTM hybrid model.
  - OpenCV: For image preprocessing (resizing, normalization).
  - NumPy: For data manipulation and preparation.
- **Dataset**: MRI images of Alzheimer's patients categorized into four stages.
- **Tools**: 
  - PyCharm: Development environment.
  - GitHub: Version control and project hosting.

---

### 2. How the Project Was Built
1. **Dataset Preparation**:
   - Collected MRI images representing four stages of Alzheimer's disease.
   - Preprocessed images to normalize pixel values and resize them to 150x150 dimensions using OpenCV.
   - Organized the data into respective folders for training, validation, and testing.

2. **Model Design**:
   - Designed a **CNN** to capture spatial features from MRI images.
   - Added an **LSTM layer** to analyze sequential patterns across the extracted features.
   - Used categorical cross-entropy as the loss function and Adam optimizer for training.

3. **Training**:
   - Split the dataset into training and validation sets.
   - Trained the model on the processed dataset, optimizing hyperparameters to improve classification accuracy.

4. **Testing and Evaluation**:
   - Evaluated the trained model on a test dataset.
   - Classified images into one of the four stages based on model predictions.

5. **Prediction Script**:
   - Developed `index2.py` to load the trained CNN-LSTM model and predict the stage of Alzheimer's for a given input image.

---

## Project Structure
```plaintext
Alzheimer_Detection_Project/
├── Dataset/                   # Dataset folder (not fully pushed to GitHub)
│   └── [subfolders for data]  # Contains MRI images
├── model/                     # Folder for saved model files
│   └── CNN_RNN_save.h5        # Pre-trained CNN-LSTM model
├── scripts/                   # Python scripts for training and prediction
│   ├── index2.py              # Script for making predictions
│   └── cnn.py                 # Script for model training
├── README.md                  # Project documentation
