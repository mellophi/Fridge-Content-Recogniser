from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import csv
import os
import cv2
from fyp_utils import CustomDataset, ImageFolderWithPaths, set_parameter_requires_grad, initialize_model, csv_to_dict, captureImage, delete_all_files, dict_to_json, predict_model
# path to the saved checkpoint
saved_path = r'/home/mello/.cache/torch/checkpoint/saved_squeezenet_120_train_0.98347_test_1.0_2020-04-21_12-42-17.pth'

#path to save the test images
path_to_save = r'D:\Dev\FYP\test_image'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 200

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


model_ft, input_size = initialize_model(3, True)
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation((-30, 30)),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

captureImage(path_to_save)

# image_datasets = ImageFolderWithPaths(r'D:\Dev\FYP\test_image', data_transforms['test'])
# testloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False)
image_datasets = CustomDataset(r'D:\Dev\FYP\test_image')
testloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter()
loaded = torch.load('saved_squeezenet_195_train_0.96758_test_1.0_2020-04-23_16-37-45.pth')
model_ft.load_state_dict(loaded['model_dict'])

predict_model(model_ft, testloader)

delete_all_files(path_to_save, 'data.json')

# for i, images in enumerate(custom_dataset_loader):
#     images = Variable(images)
#     outputs = model(images)
#     _, preds = torch.max(outputs, 1)
#     l = int(preds.cpu().numpy())
#     #get weight from get weight function
#     # weight = 1
#     # dict = {}
#     # dict[result[l]] = weight 
#     # dict_to_json(dict, 'data.json')
#     print(result[l])