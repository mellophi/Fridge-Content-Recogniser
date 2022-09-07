from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import csv
import os
import cv2
from fyp_utils import CustomDataset, set_parameter_requires_grad, initialize_model, csv_to_dict, captureImage, delete_all_files

# path to the saved checkpoint
saved_path = '/home/mello/.cache/torch/checkpoint/saved_squeezenet_195_train_0.8749_test_0.81028_2020-04-04_20-39-02.pth'

#path to save the test images
path_to_save = '/home/mello/dev/enginx/test_image'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

# Number of classes in the dataset
num_classes = 12

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 200

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

path_to_csv = 'classes.csv'

result = csv_to_dict(path_to_csv)

captureImage(path_to_save)

custom_dataset_image = CustomDataset(path_to_save)
custom_dataset_loader = torch.utils.data.DataLoader(dataset = custom_dataset_image)

# class CustomDataset(Dataset):
#     def __init__(self, img_path):
#         self.to_tensor = transforms.ToTensor()
#         image_path = []
#         for subdir, dirs, files in os.walk(img_path):
#             filepath = subdir + os.sep + file
#             if filepath.endswith('.jpg'):
#                 image_path.append(filepath)
                
#         self.image_arr = np.asarray(image_path)
#         self.data_len = len(image_path)
    
#     def __getitem__(self, index):
#         single_image_name = self.image_arr[index]
#         img_as_img = Image.open(single_image_name)
#         img_as_tensor = self.to_tensor(img_as_img)
        
#         return img_as_tensor
    
#     def __len__(self):
#         return self.data_len

# path_to_save = input("Enter path to save: ")

# from fyp_utils import captureImage
# captureImage(path_to_save)

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded = torch.load(saved_path)
model.load_state_dict(loaded['model_dict'])
model.eval()

for i, images in enumerate(custom_dataset_loader):
    images = Variable(images)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    l = int(preds.cpu().numpy())
    print(result[l])

delete_all_files(path_to_save)

