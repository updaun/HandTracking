import cv2
import numpy as np
import time
import autopy
from autopy.mouse import Button
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules.HandTrackingModule import handDetector

#####################################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7 
#####################################

pTime = 0
plocX, plocY = 0,0
clocX, clocY = 0,0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

detector = handDetector(max_num_hands=1)
wScr, hScr = autopy.screen.size()

while True:

    # Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[5][1:]
    
        # Check which fingers are up
        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR),
                        (255,0,255), 2)

        # Click
        if fingers[1:] == [0,0,0,0]:
            autopy.mouse.toggle(button=Button.LEFT, down=True)
        else:
            autopy.mouse.toggle(button=Button.LEFT, down=False)

        # Convert Coordicates
        x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
        y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

        # Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # Move Mouse
        try:
            autopy.mouse.move(wScr-x3, y3)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        except:
            print("Point out of bounds")

        
                
    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    img = cv2.flip(img, 1)
    
    # cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    # (255,0,0), 3)

    # Display
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break