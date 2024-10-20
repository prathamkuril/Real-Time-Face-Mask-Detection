# Real Time Face Mask Dectection using Convolution Neural Networks and Hare Cascade Classifier

This project consists of three Jupyter notebooks: `21.ipynb`, `22.ipynb`, and `23.ipynb`, which work together to create a mask detection model using convolutional neural networks (CNNs) and OpenCV for real-time face mask detection. Below is a brief explanation of the steps and processes involved.

## Notebooks Overview

### 1. `21.ipynb` - Data Preprocessing

This notebook handles the following tasks:
- **Loading and Labeling the Dataset**: Images of people with and without masks are loaded from the dataset directory.
- **Image Processing**: The images are converted to grayscale and resized to 100x100 pixels for uniformity. This simplifies processing and ensures that the images are the same size when fed into the neural network.
- **Handling Exceptions**: If any image processing errors occur (such as an empty source), they are caught and handled gracefully.
- **Data Normalization**: The pixel values are normalized to the range of 0 to 1 to improve the training performance of the neural network.
- **Saving Processed Data**: The preprocessed data and their corresponding labels are saved as NumPy arrays (`data.npy` and `target.npy`) for later use.

### 2. `22.ipynb` - Model Training

This notebook focuses on training the CNN model for mask detection:
- **Loading Data**: The preprocessed data saved in `21.ipynb` is loaded.
- **Model Architecture**: A convolutional neural network (CNN) is built using the Keras library, with the following layers:
  - Two convolutional layers followed by ReLU activation and max-pooling.
  - A dropout layer to prevent overfitting.
  - A fully connected dense layer followed by a softmax activation to classify the input as "mask" or "no mask."
- **Training and Validation**: The model is trained using 20 epochs, with 10% of the data held out for validation. The model checkpoints are saved after each epoch if they improve the validation loss.
- **Plotting Training History**: After training, the loss and accuracy (for both training and validation sets) are plotted for better visualization.
- **Model Evaluation**: The trained model is evaluated on a test set to calculate its performance metrics, such as accuracy.

### 3. `23.ipynb` - Real-Time Mask Detection

This notebook runs the real-time mask detection using a webcam feed:
- **Loading Pretrained Model**: The best-performing model from the previous notebook is loaded.
- **Face Detection**: OpenCV's Haar Cascade Classifier is used to detect faces in the video stream.
- **Mask Detection**: For each detected face, the model classifies whether the person is wearing a mask or not.
- **Real-Time Display**: Bounding boxes are drawn around the faces, and text is displayed to indicate whether the person is wearing a mask or not. Green indicates "mask," and red indicates "no mask."
- **Stopping the Detection**: Press the `Esc` key to stop the live feed.

## Requirements

To run this project, you'll need to have the following libraries installed:

```bash
pip install opencv-python
pip install keras
pip install tensorflow
pip install numpy
pip install matplotlib
```

## Instructions

1. **Dataset**: Place your dataset of mask and no-mask images in the specified `data_path`. Ensure that the dataset is organized in separate folders for "with mask" and "without mask."

2. **Run the Notebooks**:
   - First, execute `21.ipynb` to preprocess the data.
   - Then, run `22.ipynb` to train the CNN model.
   - Finally, use `23.ipynb` to perform real-time mask detection.

3. **Live Detection**: Ensure that your webcam is connected, and execute `23.ipynb` to start the live mask detection.

## Notes

- Ensure that OpenCV is properly installed, as the Haar Cascade Classifier for face detection is heavily reliant on OpenCV.
- The dataset must be well-organized for the program to correctly label and process images.
- The model accuracy will depend on the quality and size of the dataset used. Fine-tuning may be required for optimal results.

## Acknowledgments

This project uses Keras for deep learning and OpenCV for computer vision tasks.
