import tensorflow as tf
from keras import layers
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Define the image size for resizing
IMG_SIZE = (640, 480)

# Define the path to your dataset
DATASET_PATH = "D:\HCMUT\OEC\Innowork\OpenCV"

# Define the labels for your classes
CLASSES = ["defected", "nodefected"]

# Create an empty list to store the preprocessed images and labels
data = []
labels = []

# Loop through the dataset and preprocess the images
for category in CLASSES:
    path = os.path.join(DATASET_PATH, category)
    class_num = CLASSES.index(category)
    print(class_num)
    for img in os.listdir(path):
        try:
            # Load the image and convert it to grayscale
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            
            # Resize the image to a fixed size
            new_array = cv2.resize(img_array, IMG_SIZE)
            
            # Add an extra dimension to the image
            new_array = np.reshape(new_array, (*IMG_SIZE, 1))
            
            # Normalize the pixel values
            new_array = new_array / 255.0
            
            # Add the preprocessed image and label to the data list
            data.append(new_array)
            labels.append(class_num)
        except Exception as e:
            pass
print(labels)
# Convert the data and labels lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Convert the labels to one-hot encoding
labels = to_categorical(labels)

# Split the dataset into training, validation, and testing sets

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42)

# Define the input shape of your images
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)

# Define the number of classes
num_classes = len(CLASSES)

# Define the CNN model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),    metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10,
                    validation_data=(val_data, val_labels))
