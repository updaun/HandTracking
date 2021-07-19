from typing import overload
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

overlayList =[]

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]
drawColor = (50, 50, 50)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 800)
cap.set(4, 600)

detector = htm.handDetector(detectionCon=0.85)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)
        # tip of Index fingers
        x1, y1 = lmList[8][1:]
        # tip of Middle fingers
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            # Checking for the click
            if y1 < 100:
                # Dark Gray
                if 150<x1<200:
                    header = overlayList[0]
                    drawColor = (100, 100, 100)
                # Deep Blue
                elif 300<x1<350:
                    header = overlayList[1]
                    drawColor = (111, 6, 6)
                # Deep Green
                elif 400<x1<450:
                    header = overlayList[2]
                    drawColor = (8, 102, 5)
                # Orange
                elif 500<x1<550:
                    header = overlayList[3]
                    drawColor = (0, 69, 255)
                # Yellow
                elif 625<x1<675:
                    header = overlayList[4]
                    drawColor = (0, 255, 255)
                # Eraser 
                elif 700<x1<800:
                    header = overlayList[5]
                    drawColor = (255, 255, 255)
                
            cv2.rectangle(img, (x1, y1-20), (x2, y2+20), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

    # Setting the header image
    img[0:100, 0:800] = header
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()