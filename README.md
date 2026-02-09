# MNIST Digit Recognition Project

A comprehensive deep learning project for handwritten digit classification using Convolutional Neural Networks (CNN). This project achieves high accuracy through data augmentation and advanced model architectures.

## Project Overview

This project implements a complete machine learning pipeline for recognizing handwritten digits (0-9) from the MNIST dataset. The implementation includes data preprocessing, model building, training with augmentation, and evaluation.

## Dataset

- **Training Set**: 42,000 images with labels
- **Test Set**: 28,000 images for prediction
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

## Features

- Data visualization and exploratory analysis
- Convolutional Neural Network architecture
- Data augmentation for improved generalization
- Model performance evaluation and comparison
- Custom image testing capabilities
- Competition-ready submission generation

## Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Model evaluation and data splitting

## Model Architecture

### Baseline CNN Model
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax output layer for multi-class prediction

### Augmented Model
Enhanced version trained on augmented data with:
- Rotation range: ±10 degrees
- Width/Height shift: ±10%
- Zoom range: ±10%

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-digit-recognition.git

# Navigate to project directory
cd mnist-digit-recognition

# Install required packages
pip install -r requirements.txt
```

## Usage

### Running the Notebook

```bash
jupyter notebook minist.ipynb
```

### Training the Model

The notebook includes step-by-step cells for:
1. Loading and preprocessing data
2. Building the CNN model
3. Training with and without augmentation
4. Evaluating model performance
5. Making predictions

### Making Predictions

```python
# Load your custom image
from keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_image.png', target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
```

## Results

- **Baseline Model Accuracy**: ~98%
- **Augmented Model Accuracy**: ~99%
- **Training Time**: Approximately 15-20 minutes per model
- **Test Set Performance**: Competitive results on Kaggle leaderboard

## Project Structure

```
mnist-digit-recognition/
│
├── minist.ipynb              # Main Jupyter notebook
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── submission_augmented.csv  # Kaggle submission file
└── data/                     # Dataset directory
    ├── train.csv
    └── test.csv
```

## Data Augmentation Techniques

The project employs various augmentation strategies:
- **Rotation**: Random rotation within ±10 degrees
- **Translation**: Horizontal and vertical shifts
- **Zoom**: Random zoom in/out
- **Normalization**: Pixel value scaling to [0,1]

## Model Training Process

1. **Data Preprocessing**: Normalization and reshaping
2. **Train/Validation Split**: 80/20 split
3. **Model Compilation**: Adam optimizer with categorical cross-entropy
4. **Training**: 20 epochs with early stopping
5. **Evaluation**: Validation accuracy and loss metrics

## Performance Visualization

The notebook includes comprehensive visualizations:
- Digit distribution analysis
- Sample image displays
- Training/validation accuracy curves
- Loss curves over epochs
- Confusion matrix for predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- MNIST Dataset from Kaggle Playground Series
- TensorFlow and Keras documentation
- Data science community for best practices and techniques

## Contact

For questions or feedback, please open an issue in the repository.

## Kaggle Competition

This project was developed as part of the Kaggle Playground Series - Season 6, Episode 2.

---

**Note**: Make sure to download the dataset from Kaggle and place it in the appropriate directory before running the notebook.
