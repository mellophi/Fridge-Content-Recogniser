from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import cv2
import os
import re
import csv
from datetime import datetime

def get_datetime():
    date_pat = re.compile(
            "^(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})?")
    match = date_pat.search(str(datetime.now()))
    date, time = match.group(1), match.group(2)
    time = time.replace(':','-')
    return date, time

def captureImage(path_to_save):
    path_to_save = path_to_save.rstrip('/')
    vegename = path_to_save.split('/')[-1]
    vegecount = 0
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)

    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                img_ = cv2.resize(img_,(348,348))
                filename = os.path.join(path_to_save, '{}_{}.jpg'.format(vegename, vegecount))
                img_resized = cv2.imwrite(filename=filename, img=img_)
                print('{}_{}.jpg saved!!'.format(vegename, vegecount))
                vegecount = vegecount + 1
                os.remove('saved_img.jpg')
                break

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Biday Pitibi :)")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Biday Pitibi :)")
            cv2.destroyAllWindows()
            break

class CustomDataset(Dataset):
    def __init__(self, img_path):
        self.to_tensor = transforms.ToTensor()
        image_path = []
        for subdir, dirs, files in os.walk(img_path):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith('.jpg'):
                    image_path.append(filepath)
                
        self.image_arr = np.asarray(image_path)
        self.data_len = len(image_path)
    
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        img_as_tensor = self.to_tensor(img_as_img)
        
        return img_as_tensor
    
    def __len__(self):
        return self.data_len

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

def csv_to_dict(path_to_csv):
    df = csv.reader(open(path_to_csv))
    result = {}
    for row in df:
        key = int(row[1])
        result[key] = row[0]

    return result

def delete_all_files(path_to_save):
    mypath = "my_folder"
    for root, dirs, files in os.walk(path_to_save):
        for file in files:
            os.remove(os.path.join(root, file))

    

# path_to_save = input("Enter the path to save: ")
# captureImage(path_to_save)