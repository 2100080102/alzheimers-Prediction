# Alzheimer's Disease Detection Project

This project employs deep learning techniques to detect and classify Alzheimer's disease stages based on MRI images. By combining Convolutional Neural Networks (CNN) with Recurrent Neural Networks (RNN), the model effectively analyzes both spatial and sequential features of the images.

## Features
- **Stage Classification**: Predicts one of the following stages:
  - Mild
  - Moderate
  - Non-Demented
  - Very Mild
- **Model Architecture**: Combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for enhanced prediction accuracy.
- **Frameworks and Libraries Used**: TensorFlow for model creation, OpenCV for image preprocessing, and NumPy for data manipulation.

---

## Implementation Details

### 1. Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**:
  - TensorFlow: For building and training the CNN-RNN hybrid model.
  - OpenCV: For image loading and preprocessing.
  - NumPy: For numerical operations.
- **Dataset**: MRI images of Alzheimer’s patients categorized into different stages.
- **Tools**: 
  - PyCharm: Development environment.
  - CustomTkinter: For GUI integration (future plans, if applicable).
  - GitHub: For version control and repository management.

---

### 2. How the Project Was Built
1. **Dataset Preparation**:
   - Collected MRI image data representing four stages of Alzheimer's disease.
   - Preprocessed images using OpenCV to normalize pixel values and resize images to a consistent size (150x150).
   - Organized data into respective folders for training, validation, and testing.

2. **Model Design**:
   - Built a Convolutional Neural Network (CNN) for spatial feature extraction from MRI images.
   - Integrated an LSTM layer to analyze sequential features and patterns across the data.
   - Compiled the model with categorical cross-entropy as the loss function and Adam optimizer.

3. **Training**:
   - Split the dataset into training and validation sets.
   - Trained the model on preprocessed MRI images, adjusting hyperparameters for optimal accuracy.

4. **Testing and Evaluation**:
   - Evaluated the trained model using a test set.
   - Achieved classification accuracy and generated predictions.

5. **Prediction Script**:
   - Developed `index2.py` to load the trained model and predict the stage of Alzheimer's from a given image.

---

## Project Structure
```plaintext
Alzheimer_Detection_Project/
├── Dataset/                   # Dataset folder (not fully pushed to GitHub)
│   └── [subfolders for data]  # Contains MRI images
├── model/                     # Folder for saved model files
│   └── CNN_RNN_save.h5        # Pre-trained model
├── scripts/                   # Python scripts for training and prediction
│   ├── index2.py              # Script for making predictions
│   └── cnn.py                 # Script for model training
├── README.md                  # Project documentation
