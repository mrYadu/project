import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import create_model

# Load and preprocess data
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    categories = os.listdir(data_dir)

    for category in categories:
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(categories.index(category))

    return np.array(images), np.array(labels), categories

# Main function
if __name__ == "__main__":
    data_dir = 'data'  # Path to the data directory
    images, labels, categories = load_data(data_dir)
    images = images / 255.0  # Normalize images
    labels = to_categorical(labels)  # One-hot encode labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create and compile the model
    model = create_model((128, 128, 3), len(categories))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model.save('project.keras')  # Change this line