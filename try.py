import cv2
import pickle
import numpy as np
import cvzone

cap = cv2.VideoCapture('carPark.mp4')
width, height = 107, 48

with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

def checkParkingSpace(imgPros):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        # cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (255, 0, 255), 2)

        img_crop = imgPros[y:y+height, x:x+width]
        # cv2.imshow(str(x*y), img_crop)
        count = cv2.countNonZero(img_crop)
        cvzone.putTextRect(img, str(count), (x, y+height-5), scale=1, thickness=1, offset=0, colorR=(0, 0, 255))

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 255, 0))

while True:
    success, img = cap.read()
    if not success:
        cap = cv2.VideoCapture('carPark.mp4')
        continue
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (255, 0, 255), 2)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    checkParkingSpace(imgDilate)
    cv2.imshow("Images", img)
    # cv2.imshow("Image", imgDilate)
    cv2.waitKey(10)
