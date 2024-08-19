import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Load the dataset
def load_fer2013():
    data = pd.read_csv(r"C:\Users\ACER\Desktop\Suhas\Internship\CodeClause\Emotion_Detection\fer2013.csv")
    pixels = data['pixels'].tolist()
    
    # Convert the pixels from string to a numpy array
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48)
        face = face.astype('float32')
        faces.append(face)
    
    # Convert faces to numpy array and normalize
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # Add the channel dimension
    faces /= 255.0  # Normalize

    # Convert emotion labels to categorical one-hot encoding
    emotions = pd.get_dummies(data['emotion']).to_numpy()

    return faces, emotions

# Load the dataset
X_train, y_train = load_fer2013()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=25, validation_data=(X_test, y_test))

# Save the model
model.save('emotion_detection_model.h5')
