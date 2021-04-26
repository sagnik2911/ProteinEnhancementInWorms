import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import numpy as np
import matplotlib.pyplot as plt
import PIL


class WormImagesDataset(Dataset):

    def __init__(self, input_dir_name, output_dir_name, transform):
        self.input_dir_name = input_dir_name
        self.output_dir_name = output_dir_name
        self.input_images = os.listdir(input_dir_name)
        self.output_images = os.listdir(output_dir_name)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image_input_name = os.path.join(self.input_dir_name, self.input_images[idx])
        image_output_name = os.path.join(self.output_dir_name, self.output_images[idx])
        image_input = PIL.Image.open(image_input_name)
        image_output = PIL.Image.open(image_output_name)
        if self.transform is not None:
            image_input = self.transform(image_input)
            image_output = self.transform(image_output)
        image_input_asnp = np.asarray(image_input)
        image_output_asnp = np.asarray(image_output)

        # image_output_asnp_filtered = image_output_asnp[:, :, 0:3]
        # image_input_asnp = rgb2gray(image_input_asnp)
        # image_output_asnp = rgb2gray(image_output_asnp_filtered)
        image_input_asnp = (image_input_asnp + 128) / 255
        image_output_asnp = (image_output_asnp + 128) / 255

        x = torch.from_numpy(image_input_asnp).float()
        y = torch.from_numpy(image_output_asnp).float()

        # print(str(image_input_name) + " : " + str(x.shape))
        # print(image_output_name + " : " + str(y.shape))
        return x, y
