from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import numpy as np
import cv2
import os
import re
import csv
from datetime import datetime
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import statistics

def get_datetime():
    date_pat = re.compile(
            "^(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})?")
    match = date_pat.search(str(datetime.now()))
    date, time = match.group(1), match.group(2)
    time = time.replace(':','-')
    return date, time

def captureImage(path_to_save):
    path_to_save = path_to_save.rstrip(os.sep)
    vegename = path_to_save.split(os.sep)[-1]
    vegecount = 0
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(1)

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
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
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

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1,1))
    model_ft.num_classes = num_classes
    input_size = 224
    return model_ft, input_size

def csv_to_dict(path_to_csv):
    df = csv.reader(open(path_to_csv))
    result = {}
    for row in df:
        key = int(row[1])
        result[key] = row[0]

    return result

def delete_all_files(path_to_save, json_filename='data.json'):
    # removing the captured image fro prediction after the prediction is done
    for root, dirs, files in os.walk(path_to_save):
        for file in files:
            os.remove(os.path.join(root, file))

    # removing the json file after 1 day
    if os.path.exists(json_filename):
        data = json.load(open(json_filename))
        date, time = get_datetime()
        for key in data.keys():
            if key != date:
                os.remove(json_filename)
        
def dict_to_csv(dict, csv_filename):
    with open(csv_filename, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value]) 

# saving the json in the follwing format {date: data for that date}
def dict_to_json(captured_data, filename):
    date, time = get_datetime()
    if os.path.exists(filename):
        data = json.load(open(filename))
        temp = data[date]
        if not temp:
            data = {date:[]}
        data[date].append(captured_data)
        j = json.dumps(data)
        with open (filename, 'w', newline="") as f:
            f.write(j)
            f.close()
    else:
        data = {date:[]}
        data[date].append(captured_data)
        j = json.dumps(data)
        with open (filename, 'w', newline="") as f:
            f.write(j)
            f.close()


def json_to_dict(filename):
    return json.load(open(filename))  

def predict_model(model, dataloaders, num_epochs=1, is_inception=False, save_interval=5):
    result = csv_to_dict('classes.csv')
    for epoch in range(num_epochs):
        model.eval()
        with torch.no_grad():
            for inputs in dataloaders:
#                 inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                l = int(preds.cpu().numpy())
                # get weight from function
                weight = float(input("Enter the weight of the vegetable: "))
                print(result[l])  

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def update_threshold(fetched_data_temp, ref_thresh, ref_temp):
    list_keys = list(fetched_data_temp.keys())
    for key in list_keys:
        fetched_data_temp_child = ref_temp.child(key).get()
    list_temp_data_dict = list(fetched_data_temp_child.values())
    avg_dict = {'Bittergourd':[], 'Potato':[], 'Onion':[]}
    for d in list_temp_data_dict:
        for key, value in d.items():
            avg_dict[key].append(value)
    
    for key, value in avg_dict.items():
        if value:
            ref_thresh.update({
                key:statistics.mean(value)
            })


def upload_files(vegetable_name, vegetable_weight):
    # connention to the database
    cred = credentials.Certificate('firebase-sdk.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL':'https://firepy-5998e.firebaseio.com/'
    })
    date, time = get_datetime()

    #refences to the database 
    ref_master = db.reference('/Master') 
    ref_temp = db.reference('/Temp') 
    ref_temp_child = ref_temp.child(date) #Master/date reference
    ref_thresh = db.reference('/Threshold')

    # fetching all data
    fetched_data_threshold = ref_thresh.get()
    fetched_data_temp = ref_temp.get()
    fetched_data_master = ref_master.get()
    fetched_data_temp_child = ref_temp_child.get()

    # deleted the values if it is not in the same date
    if ref_temp.get():
        for key, value in ref_temp.get().items():
            if key != date:
                update_threshold(fetched_data_temp, ref_thresh, ref_temp)
                ref_temp.delete()


    # adding old data to the history table (Temp)
    ref_temp.child(date).push({
        vegetable_name:float(fetched_data_master[vegetable_name])
    })
    
    # updating old data to the new data in master table(Master)
    new_value = float(fetched_data_master[vegetable_name]) + vegetable_weight
    ref_master.update({
        vegetable_name:new_value
    })
    