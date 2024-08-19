import imagej
import os
import pandas as pd
import sys

# argumnets
file_path = str(sys.argv[1])
panel= str(sys.argv[2])
file_name=os.path.basename(file_path).split('/')[-1].split('.')[0]

# Get channel names
panel = pd.read_csv(panel)
channel_names = panel['name'].tolist()
channel_names = [string.replace(' ', '_') for string in channel_names]

# Initialize ImageJ
ij = imagej.init()

# Load the multichannel TIFF file
img = ij.io().open(file_path)

# Get the dimensions of the image (e.g., width, height, channels)
channels = img.getChannels()

# Convert to numpy array for easy manipulation
img_np = ij.py.from_java(img)

# Define channel names (these should match the channels in your TIFF)

# Ensure the number of channel names matches the channels in the image
if len(channel_names) != channels:
    raise ValueError("Number of channel names must match the number of channels in the image.")

# Create a directory to save the single-channel files
output_dir = os.path.join('single_channel_images',file_name)
os.makedirs(output_dir, exist_ok=True)

# Loop through each channel and save as individual file
for i in range(channels):
    # Select the channel
    channel_img_np = img_np[:, :, i]
    
    # Convert numpy array back to ImagePlus
    channel_img = ij.py.to_java(channel_img_np)
    
    # Save the file with the channel name in the filename
    output_file_path = os.path.join(output_dir, f'{file_name}_{channel_names[i]}.tiff')
    ij.io().save(channel_img, output_file_path)

print("Single-channel images saved successfully.")
