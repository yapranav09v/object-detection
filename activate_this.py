#mobile vs cover

import cvzone
import cv2

cap = cv2.VideoCapture(0)

myclassifier = cvzone.Classifier('mobc/keras_model.h5', 'mobc/labels.txt')
fpsReader = cvzone.FPS()

while True:
    _, img = cap.read()
    predictions, index = myclassifier.getPrediction(img, scale=1)
    print(predictions, index)
    fps, img = fpsReader.update(img, pos=(450,50))
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

