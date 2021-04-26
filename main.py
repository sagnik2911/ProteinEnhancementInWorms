# For plotting
import sys
import os
import cv2
import PIL
# For conversion
import torchvision

from cnn_protein_enhacer import CNNProteinEnhancer
from helper import AverageMeter
from custom_dataset_from_folder import WormImagesDataset
from torchvision import transforms
from scipy import misc
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
# For everything
import torch.utils.data
import torch.nn as nn
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
# For utilities
import time
import image_slicer
from image_slicer import join
import imageio

best_losses = float("inf")
use_gpu = torch.cuda.is_available()


def trainCNN(train_loader, model, criterion, optimizer, epoch):
    print('Starting Training Epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, losses = AverageMeter(), AverageMeter()
    end = time.time()

    for i, (x, y) in enumerate(train_loader):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        # x = x.permute(0, 3, 1, 2)
        # y = y.permute(0, 3, 1, 2)
        # Use GPU if available
        if use_gpu:
            x, y = x.cuda(), y.cuda()

        # Run Forward Pass
        output_y = model(x)
        # print(y)
        # print(output_y)
        # break
        loss = criterion(output_y, y)
        losses.update(loss.item(), x.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do backward and forward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))
    return losses.avg


def train():
    global best_losses, use_gpu
    model = CNNProteinEnhancer()
    if use_gpu:
        model.cuda()
        print('Loaded model onto GPU.')

    criterion = nn.BCELoss().cuda() if use_gpu else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    transforms_preprocess = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    transforms_for_augmentation = [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1),
                                   transforms.Compose(
                                       [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1)])]

    dataset = WormImagesDataset('input_images', 'output_images', transform=transforms_preprocess)

    augmenteddataset = dataset
    # for transform in transforms_for_augmentation:
    #     augmenteddataset = torch.utils.data.ConcatDataset(
    #         [augmenteddataset, WormImagesDataset('input_images', 'output_images', transform)])

    train_loader = torch.utils.data.DataLoader(augmenteddataset, batch_size=16, shuffle=True)
    print("Loaded Training Data : " + str(len(train_loader)))

    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()
    print("Running Training of Encoder")
    epochs = 100
    loss_train = []
    for epoch in range(epochs):
        losses = trainCNN(train_loader, model, criterion, optimizer, epoch)
        loss_train.append(losses)
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
    print('Completed Execution. Best loss: ', best_losses)

    # sample = iter(train_loader)
    # images = next(sample)
    # x, y = images
    # print(y)
    # print(x)
    # x = x * 255 - 128
    # y = y * 255 - 128
    # f = plt.figure()
    # f.add_subplot(1, 2, 1)
    # plt.imshow(np.uint8(x[0]), cmap='gray')
    # f.add_subplot(1, 2, 2)
    # plt.imshow(np.uint8(y[0]), cmap='gray')
    # plt.show(block=True)


def output(model):
    output_dir = 'test_images'
    img_name = 'PL13_C04_S2_C2_ch00.tif'
    img = os.path.join(output_dir, img_name)
    num_tiles = 16
    tiles = image_slicer.slice(img, num_tiles, save=True)

    for i in range(0, num_tiles):
        img = tiles[i].image

        img_asnp = np.asarray(img)
        img_astensor = torch.from_numpy(img_asnp).float()
        img_astensor = img_astensor.unsqueeze(0)
        img_astensor = img_astensor.permute(0, 3, 1, 2)
        out = model(img_astensor)
        out = out.squeeze(0)
        out = out.permute(1, 2, 0)
        print(out.shape)
        print(img_astensor)
        print(out)
        out_asnp = out.detach().numpy()
        print(np.shape(out_asnp))
        print(tiles[i].filename)
        imgout = Image.fromarray((out_asnp * 255).astype(np.uint8))
        imgout.save(tiles[i].filename[0:-4] + 'Out' + '.png')
        tiles[i].image = out_asnp
        # tile.image = Image.open(tile.filename)

    imageOutput = join(tiles)
    imageOutput.save('Out' + img_name)


def test():
    pretrained_model = 'model-epoch-99-losses-0.692.pth'
    model = CNNProteinEnhancer()
    model.state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
    print('Model Loaded from memory')
    output(model)


if __name__ == '__main__':
    testmode = sys.argv[1]
    print('The network is running in Test Mode') if int(testmode) == 1 else print(
        'The network is running in Train Mode')
    if int(testmode) == 1:
        test()
    else:
        train()
