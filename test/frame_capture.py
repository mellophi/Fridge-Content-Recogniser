import cv2
import os
import re
from datetime import datetime

def get_datetime():
    date_pat = re.compile(
            "^(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})?")
    match = date_pat.search(str(datetime.now()))
    date, time = match.group(1), match.group(2)
    time = time.replace(':','-')
    return date, time

print("--------------------------------")
print("Hello this code is to capture images")
print("q: For quitting tha capture ")
print("s: For saving the image")
print("Enjoy :)")
print("-------------------------------")

path_to_save = input("Enter the path to save: ")
path_to_save = path_to_save.rstrip(os.sep)
vegename = path_to_save.split(os.sep)[-1]
vegecount = int(input("Enter count to start from: "))
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(1)

frame_rate = webcam.get(5) #frame rate

while True:
    try:
        frame_id = webcam.get(1) #current frame number
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if frame_id % math.floor(frame_rate) == 0:
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