import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# Load IMDB-WIKI Dataset (Assuming you have it downloaded)
DATASET_PATH = "path_to_dataset"
metadata = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv")) 
# Preprocessing Function
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    return image

# Load Images and Labels
images = []
ages = []

for index, row in metadata.iterrows():
    img_path = os.path.join(DATASET_PATH, row['image_path'])
    if os.path.exists(img_path):
        images.append(preprocess_image(img_path))
        ages.append(row['age'])

images = np.array(images)
ages = np.array(ages)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# Define CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='linear')  # Age is a regression task
])

# Compile Model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save Model
model.save("age_detection_model.h5")

# Plot Training Results
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.show()
