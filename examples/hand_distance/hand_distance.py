import cv2
import math
import numpy as np
import cvzone

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules import HandDetector

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find Function
# x : raw distance  y : value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# 2차원 그래프 추적
coff = np.polyfit(x, y, 2) # AX^2 + BX + C

# Loop
while True:
    success, img = cap.read()

    hands = detector.findHands(img, draw=False)

    if hands:
        lmList = hands[0]['lmList']
        
        x,y,w,h = hands[0]['bbox']

        x1, y1 = lmList[5][1:3]
        x2, y2 = lmList[17][1:3]

        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        A, B, C = coff
        # distanceCM = int(A*distance**2 + B*distance + C)
        distanceCM = int(A*distance**2 + B*distance + C/2)
        print(distance, distanceCM)

        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 3)

        cvzone.putTextRect(img, f'{distanceCM} cm', (x+5,y-10))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.waitKey(1)
