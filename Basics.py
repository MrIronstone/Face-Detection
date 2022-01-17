import cv2
import numpy as np
import face_recognition

font = cv2.FONT_HERSHEY_SIMPLEX

# training
imgElon = face_recognition.load_image_file('Berk Yuz/berk yuz 1.png')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# test
imgTest = face_recognition.load_image_file('Berk Yuz/berk yuz 2.png')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# intentionally different person
imgHusam = face_recognition.load_image_file('Gozde Staj/hüsam.png')
imgHusam = cv2.cvtColor(imgHusam, cv2.COLOR_BGR2RGB)

# training
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# draw a rectangle
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# test
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
# draw a rectangle
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# intentionally different person
faceLocIDP = face_recognition.face_locations(imgHusam)[0]
encodeIDP = face_recognition.face_encodings(imgHusam)[0]
# draw a rectangle
cv2.rectangle(imgHusam, (faceLocIDP[3], faceLocIDP[0]), (faceLocIDP[1], faceLocIDP[2]), (255, 0, 255), 2)

# results
results1 = face_recognition.compare_faces([encodeElon], encodeTest)
faceDistanceBetweenTrainingAndTest = face_recognition.face_distance([encodeElon], encodeTest)
print("Same Person?: ", results1, ", Face Distance is : ", faceDistanceBetweenTrainingAndTest)

results2 = face_recognition.compare_faces([encodeElon], encodeIDP)
faceDistanceBetweenTrainedAndIDP = face_recognition.face_distance([encodeElon], encodeIDP)
print("Same Person?: ", results2, ", Face Distance is : ", faceDistanceBetweenTrainedAndIDP)

cv2.imshow('Gozde Training', imgElon)
cv2.putText(imgTest, str(results1), (25, 25), font, 1, (0, 255, 0), 2)
cv2.imshow('Gozde Test', imgTest)

cv2.putText(imgHusam, str(results2), (25, 25), font, 1, (0, 255, 0), 2)
cv2.imshow("Intentionally Different Person aka Husam", imgHusam)

cv2.waitKey(0)
