from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pickle

def checkCarParkingSpace():
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        space_occupied = False
        for car_box in car_boxes:
            cx, cy = car_box[0] + (car_box[2] - car_box[0]) // 2, car_box[1] + (car_box[3] - car_box[1]) // 2
            if x < cx < x + width and y < cy < y + height:
                space_occupied = True
                break
        if space_occupied:
            color = (0, 0, 255)
            thickness = 2
        else:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter += 1
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        if abs(cx - (x + width // 2)) > threshold or abs(cy - (y + height // 2)) > threshold:
            cvzone.putTextRect(img, "Wrong Position", (x, y - 10), scale=0.6, thickness=1, offset=3)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                        thickness=5, offset=20, colorR=(0, 255, 0))

cap = cv2.VideoCapture("carPark.mp4")
width, height = 107, 48
threshold = 30
car_boxes = []

with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

model = YOLO("Models/best.pt")

classNames = ["car"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf > 0.9:
                # cvzone.cornerRect(img, (x1, y1, w, h))
                #
                # cls = int(box.cls[0])
                #
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                cx, cy = x1+w//2, y1+h//2
                cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)
                car_boxes.append((x1, y1, x2, y2))

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (255, 0, 255), 2)

    checkCarParkingSpace()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

