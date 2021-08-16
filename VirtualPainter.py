from typing import overload
import cv2
import numpy as np
import os
import HandTrackingModule as htm
# from datetime import datetime
# import pytz

###################################
brushThickness = 15
eraserThickness = 50
###################################

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)

overlayList =[]

img_counter = 1

# time_zone = pytz.timezone('Asia/Seoul')

# now = datetime.now(time_zone)

# current_time = now.strftime("%H%M")


for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

# default color
drawColor = (230, 230, 230)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 848, 3), np.uint8)
whiteCanvas = imgCanvas + 255
imgInv = np.zeros((480, 848, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    # print(img.shape)
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
            xp, yp = 0, 0
            print("Selection Mode")
            # Checking for the click
            if y1 < 100:
                # Dark Gray -> while
                if 200<x1<250:
                    header = overlayList[0]
                    drawColor = (230, 230, 230)
                # Deep Blue
                elif 325<x1<375:
                    header = overlayList[1]
                    drawColor = (122, 6, 6)
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
                elif 725<x1<840:
                    header = overlayList[5]
                    drawColor = (255, 255, 255)
                
            cv2.rectangle(img, (x1, y1-20), (x2, y2+20), drawColor, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (255,255,255):
                cv2.line(img, (xp, yp), (x1,y1), (0,0,0), eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), (0,0,0), eraserThickness)
                cv2.line(whiteCanvas, (xp, yp), (x1,y1), (255,255,255), eraserThickness)
                cv2.circle(img, (x1, y1), int(eraserThickness/2)+2, (230,230,230), cv2.FILLED)

            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                cv2.line(img, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)
                cv2.line(whiteCanvas, (xp, yp), (x1,y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # Setting the header image
    img[0:100, 0:848] = header
    # blend img
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("whiteCanvas", whiteCanvas)
    # cv2.imshow("ImgInv", imgInv)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', whiteCanvas)
        print("Save Canvas Successfully")
        break

    # Spacebar 또는 Return 누르면 whiteCanvas 저장
    if cv2.waitKey(1) & 0xFF == 32 or cv2.waitKey(1) & 0xFF == 13:
        cv2.imwrite(f'./output_image/canvas_{img_counter}.png', whiteCanvas)
        print("Save Canvas Successfully")
        img_counter += 1

cv2.destroyAllWindows()



