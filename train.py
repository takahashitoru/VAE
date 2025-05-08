import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import cv2

#self-made
import myvae
import mytool

################################################################
# Set parameters
################################################################

RAND_SEED = 0
DIR_DATASET = "./dataset"

#DATASET = "caribou"
#DATASET = "nessie"
DATASET = "cat"

#LABEL_LIST = [11] # when DATASET="caribou"
#LABEL_LIST = [12, 13] # when DATASET="nessie"
LABEL_LIST = [17, 18] # when DATASET="cat"

TRAIN_SIZE_RATIO = 0.8
#BATCH_SIZE = 100
BATCH_SIZE = 200

Z_DIM = 2

NUM_EPOCHS = 200
LEARNING_RATE_EXP = -3

LEARNING_RATE = 10.0 ** LEARNING_RATE_EXP
DIR_SAVE = "save_" + DATASET + "_bs" + str(BATCH_SIZE) + "_ep" + str(NUM_EPOCHS) + "_lr" + str(LEARNING_RATE_EXP)

IMAGE_SIZE = 64

################################################################
# Use double-precision floating point numbers
################################################################

torch.set_default_dtype(torch.float64) # torch.float32 by default

################################################################
# Initialization of the random number generator
################################################################

print("# RAND_SEED=", RAND_SEED)

torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)

################################################################
# Creation of the dataset
################################################################

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('L')

        label = img_path.split('/')[-1].split('_')[0] # Extract the label from the filename; the label is the token before "_"

        if self.transform:
            image = self.transform(image)

        return image, int(label)

print("# DIR_DATASET=", DIR_DATASET)
print("# DATASET=", DATASET)
print("# LABEL_LIST=", LABEL_LIST)
print("# TRAIN_SIZE_RATIO=", TRAIN_SIZE_RATIO)
print("# BATCH_SIZE=", BATCH_SIZE)

################################################################
# Split the dataset for training and validation 
################################################################

all_data = ImageDataset(os.path.join(DIR_DATASET, DATASET), transform=transforms.ToTensor())
train_size = int(len(all_data) * TRAIN_SIZE_RATIO)
val_size = len(all_data) - train_size
print("# train data size=", train_size)
print("# train iteration number per epoch=", train_size // BATCH_SIZE)
print("# validation data size=", val_size)
print("# validation iteration number per epoch=", val_size // BATCH_SIZE)

if train_size % BATCH_SIZE != 0:
    print("# train_size=", train_size, "should be divisible by BATCH_SIZE. Change BATCH_SIZE=", BATCH_SIZE, "or TRAIN_SIZE_RATIO=", TRAIN_SIZE_RATIO, ". Exit.")
    sys.exit()

train_data, val_data = random_split(all_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = False)

images, labels = next(iter(train_loader))
print("# images.size=", images.size()) # Check the number of images in the training dataset
print("# first 10 labels in the training dataset:", labels[:10])

################################################################
# Learning
################################################################

print("# Z_DIM=", Z_DIM)
print("# NUM_EPOCHS=", NUM_EPOCHS)
print("# LEARNING_RATE=", LEARNING_RATE)
print("# DIR_SAVE=", DIR_SAVE)
print("# IMAGE_SIZE=", IMAGE_SIZE)

if mytool.create_or_reset_directory(DIR_SAVE) == False:
    raise ValueError(f"Failed to create the directory {DIR_SAVE}.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import myvae
model = myvae.VAE64(Z_DIM).to(device)

optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss_epoch = 0.0
    for i, (x, labels) in enumerate(train_loader):
        input = x.to(device).view(-1, IMAGE_SIZE * IMAGE_SIZE)
        output, z, ave, log_dev = model(input)
        loss = model.criterion(output, input, ave, log_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item() * x.size(0)

        history["ave"].append(ave.detach().cpu())
        history["log_dev"].append(log_dev.detach().cpu())
        history["z"].append(z.detach().cpu())
        history["labels"].append(labels.detach().cpu())

        if (i + 1) % 50 == 0:
            print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Step: {i+1}/{len(train_loader)}, Train Loss: {loss: 0.4f}')

    avg_train_loss = train_loss_epoch / len(train_data)
    history["train_loss"].append(avg_train_loss)
    print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Average Train Loss: {avg_train_loss: 0.4f}')

    model.eval()
    val_loss_epoch = 0.0
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            input = x.to(device).view(-1, IMAGE_SIZE * IMAGE_SIZE)
            output, z, ave, log_dev = model(input)
            loss = model.criterion(output, input, ave, log_dev)
            val_loss_epoch += loss.item() * x.size(0)

        avg_val_loss = val_loss_epoch / len(val_data)
        history["val_loss"].append(avg_val_loss)
        print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Average Validation Loss: {avg_val_loss: 0.4f}')

    scheduler.step()

################################################################
# Save the final model
################################################################

file_model = os.path.join(DIR_SAVE, "model.pt")
print("# file_model=", file_model)

save_info = {
    'dataset': DATASET,
    'image_size': IMAGE_SIZE,
    'z_dim': Z_DIM,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'rand_seed': RAND_SEED,
    'model_state_dict': model.state_dict()
}
torch.save(save_info, file_model)

################################################################
# Plot the training and validation losses against epoch
################################################################

plt.figure(figsize=(10, 6))
epochs = range(1, NUM_EPOCHS + 1)
plt.plot(epochs, history["train_loss"], label='Train loss')
plt.plot(epochs, history["val_loss"], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(DIR_SAVE, "OUTPUT_train_val_loss_per_epoch.png"))
plt.savefig(os.path.join(DIR_SAVE, "OUTPUT_train_val_loss_per_epoch.eps"))

print("Training finished. Model and plots saved.")

output_text_filename = os.path.join(DIR_SAVE, "OUTPUT_train_val_loss_per_epoch.txt")

with open(output_text_filename, 'w') as f:
    f.write("epoch train_loss validation_loss\n") # Header

    for i in range(NUM_EPOCHS):
        epoch = i + 1
        train_loss = history["train_loss"][i]
        val_loss = history["val_loss"][i]
        f.write(f"{epoch} {train_loss} {val_loss}\n")

print(f"Saved the training and validation losses in {output_text_filename}.")

################################################################
# Plot latent variables, calculate and display statistics, and save statistics
################################################################

cmap = plt.cm.get_cmap('viridis', len(LABEL_LIST)) # Create a colormap
plt.figure(figsize=(8, 8))
model.eval() # Set the model to evaluation mode
latent_points_by_label = {}

with torch.no_grad():
    for i, target_label in enumerate(LABEL_LIST):
        latent_points = []
        label_dataset = ImageDataset(os.path.join(DIR_DATASET, DATASET), transform=transforms.ToTensor())
        label_loader = DataLoader(
            [item for item in label_dataset if item[1] == target_label],
            batch_size=len([item for item in label_dataset if item[1] == target_label]),
            shuffle=False
        )

        for data, _ in label_loader:
            if data.numel() > 0:
                input = data.to(device).view(-1, IMAGE_SIZE * IMAGE_SIZE)
                _, z, _, _ = model(input)
                latent_points.append(z.cpu().numpy())

        if latent_points:
            latent_points_np = np.concatenate(latent_points, axis=0)
            latent_points_by_label[target_label] = latent_points_np
            plt.scatter(latent_points_np[:, 0], latent_points_np[:, 1], color=cmap(i), label=f'Label {target_label}')

plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.title('Latent Space Visualization for All Specified Labels')
plt.legend()
plt.grid(True)
output_filename_plot = os.path.join(DIR_SAVE, "OUTPUT_z_plot.png")
output_filename_plot_eps = os.path.join(DIR_SAVE, "OUTPUT_z_plot.eps")
plt.savefig(output_filename_plot)
plt.savefig(output_filename_plot_eps)
#plt.show()

print(f"Latent variables for all specified labels plotted on a single figure and saved in {output_filename_plot} and {output_filename_plot_eps}")


################################################################
# Calculate and display the mean and standard deviation of latent variables per label
# and save the statistics to text files
################################################################

print("\nLatent Variable Statistics per Label (Extended):")
for label, latent_points in latent_points_by_label.items():
    if latent_points.size > 0:
        mean_z = np.mean(latent_points, axis=0)
        std_z = np.std(latent_points, axis=0)
        mean_str = " ".join(f"{m:.6f}" for m in mean_z)
        std_str = " ".join(f"{s:.6f}" for s in std_z)
        output_line = f"{mean_str} {std_str}"
        print(f"label: {label}, {output_line}")

        output_stats_filename = os.path.join(DIR_SAVE, f"OUTPUT_z_stats_{label}.txt")
        with open(output_stats_filename, 'w') as f:
            f.write(output_line + "\n")
        print(f"Saved extended latent variable statistics for label {label} to {output_stats_filename}")

        # Save latent variables per label to separate files
        output_latent_filename = os.path.join(DIR_SAVE, f"OUTPUT_z_plot_{label}.txt")
        with open(output_latent_filename, 'w') as f:
            for row in latent_points:
                z_str = " ".join(f"{val:.6f}" for val in row)
                f.write(f"{z_str} {label}\n")
            print(f"Saved latent variables for label {label} to {output_latent_filename}")
        
    else:
        print(f"No latent points found for label {label} to calculate statistics.")
