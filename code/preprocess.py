import os
import cv2
import numpy as np

# Define dataset path and categories
data_dir = "/Users/eyaibrahim/braintumourdetection/dataset/Brain_Tumor_Dataset"  # Corrected path
categories = ["positive", "negative"]  # Folder names within the Brain_Tumor_Dataset folder

data = []
labels = []

# Load images, resize, and label them
for category in categories:
    path = os.path.join(data_dir, category)  # Full path to the 'positive' or 'negative' folder
    print(f"Looking for images in: {path}")  # Debugging print
    label = 0 if category == "positive" else 1  # Assign labels: positive=0, negative=1
    
    # Check if the folder exists
    if os.path.exists(path):
        for img in os.listdir(path):  # Loop through the images in the folder
            try:
                # Construct the image path and read it
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                img_resized = cv2.resize(img_array, (128, 128))  # Resize to 128x128
                
                data.append(img_resized)  # Add resized image to the data list
                labels.append(label)  # Add corresponding label (0 or 1)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    else:
        print(f"Directory {path} not found!")

# Convert lists to numpy arrays and normalize pixel values
data = np.array(data).reshape(-1, 128, 128, 1) / 255.0  # Normalize pixel values and reshape for CNN
labels = np.array(labels)

# Save the processed data as numpy files for easy access later
np.save("data.npy", data)  # Save the image data
np.save("labels.npy", labels)  # Save the corresponding labels

print("Data preprocessing complete. Data and labels saved as .npy files.")
