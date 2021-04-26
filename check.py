import os
import PIL
import numpy as np
import torch
from PIL import Image

def main():
    test_dir = 'test_images'
    test_images = os.listdir(test_dir)
    f = open("output.txt", "w")

    for idx in range(len(test_images)):
        image_test_name = os.path.join(test_dir, test_images[idx])
        image_test = PIL.Image.open(image_test_name)
        image_test_asnp = np.asarray(image_test)
        image_test_astorch = torch.from_numpy(image_test_asnp).float()
        #if image_test_astorch.shape == [512,512,3] :
        f.write(str(image_test_name) + " : " + str(image_test_astorch.shape) + " \n ")


    f.close()

if __name__ == '__main__':
    main()
