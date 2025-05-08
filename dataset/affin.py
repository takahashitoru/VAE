import os
import shutil
import cv2
import numpy as np
import random

def create_or_reset_directory(directory_path):
    """
    Checks if the specified directory exists.
    If it exists, it asks the user whether to delete it.
    If the user agrees, it deletes the directory and creates a new one.
    If the directory does not exist, it creates a new one.

    Args:
        directory_path (str): The path of the directory to create or reset.

    Returns:
        bool: True if a new directory was created or an existing one was reset,
              False if the existing directory was kept or an error occurred.
    """
    if os.path.exists(directory_path):
        response = input(f"Directory '{directory_path}' already exists. Do you want to delete it and create a new one? (y/n): ").lower()
        if response == 'y':
            try:
                shutil.rmtree(directory_path)
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' has been deleted and a new one created.")
                return True  # New directory created successfully
            except OSError as e:
                print(f"Error: Failed to delete directory '{directory_path}': {e}")
                return False # Failed to delete
        else:
            print(f"Directory '{directory_path}' will be kept as is.")
            return False # Existing directory kept
    else:
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' has been created.")
            return True  # New directory created successfully
        except OSError as e:
            print(f"Error: Failed to create directory '{directory_path}': {e}")
            return False # Failed to create


def random_any(a, b):
    return a + (b - a) * random.random()

def shear_X(image, shear):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear / h * (h - src[:,1])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))

def shear_Y(image, shear):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += (shear / w * (w - src[:,0])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))

def shrink_X(image, ratio):
    height, width = image.shape
    new_width = int(width * ratio)
    b = int((width - new_width) / 2.0)
    image = cv2.copyMakeBorder(image, 0, 0, b, b, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(image, (height, width))

def shrink_Y(image, ratio):
    height, width = image.shape
    new_height = int(height * ratio)
    b = int((height - new_height) / 2.0)
    image = cv2.copyMakeBorder(image, b, b, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(image, (height, width))

def affin(input_path, output_path, num_transform=8, threshold_binary=1, shear_max=200, resize_min=0.7, shrink_min=0.7, epsilon_max=0.02, output_size=64):
    """
    Apply an affin transform to an input image specified by
    "input_path", which is supposed to be an RGBA PNG file, randomly
    and the transformed image is saved into the directory of "output_path".
    """
    
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED) # input file is supposed to be RGBA

    img = cv2.resize(img, (1024, 1024)) # upsampling

    pad = int(shear_max / 2)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, threshold_binary, 255, cv2.THRESH_BINARY)

    binary_deformed = binary.copy()

    for i in range(num_transform):
        choice = random.randrange(0, 5)
        if choice == 0:
            binary_deformed = shear_X(binary, shear_max * random_any(-1.0, 1.0))
        elif choice == 1:
            binary_deformed = shear_Y(binary_deformed, shear_max * random_any(-1.0, 1.0))
        elif choice == 2:
            binary_deformed = shrink_X(binary_deformed, random_any(shrink_min, 1.0))
        elif choice == 3:
            binary_deformed = shrink_Y(binary_deformed, random_any(shrink_min, 1.0))
        else:
            pass

    binary_deformed = cv2.bitwise_not(binary_deformed)
    binary_deformed = cv2.resize(binary_deformed, (output_size, output_size))
    cv2.imwrite(output_path, binary_deformed)


def generate_dataset(dataset_name, input_files, input_labels, size_dataset = 4000):
    if create_or_reset_directory(dataset_name):
        for i in range(size_dataset):
            for input_file, input_label in zip(input_files, input_labels):
                output_file = os.path.join(dataset_name, f"{input_label}_{i:04d}.png")
                print("output_file=", output_file)
                affin(input_file, output_file)
    
if __name__ == "__main__":
    
    random.seed(1234)

    """
    dataset_name = "caribou"
    input_files = ["caribou.png"]
    input_labels = ["11"]
    generate_dataset(dataset_name, input_files, input_labels, size_dataset = 4000)

    dataset_name = "nessie"
    input_files = ["nessie1.png", "nessie2.png"]
    input_labels = ["12", "13"]
    generate_dataset(dataset_name, input_files, input_labels, size_dataset = 4000)
    """

    dataset_name = "cat"
    input_files = ["cat1.png", "cat2.png"]
    input_labels = ["17", "18"]
    generate_dataset(dataset_name, input_files, input_labels, size_dataset = 4000)
