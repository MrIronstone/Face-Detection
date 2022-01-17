import cv2
import numpy as np
import face_recognition
import os

path = "Attedance Project"
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    