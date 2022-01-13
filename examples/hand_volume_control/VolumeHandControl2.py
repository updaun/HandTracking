import cv2
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules.HandTrackingModule import handDetector

##############################################
wCam, hCam = 640, 480
##############################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

cTime = 0
pTime = 0

detector = handDetector(max_num_hands=1, min_detection_confidence=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

area = 0
colorVol = (255, 0, 0)

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        # Filter based on size
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])//100

        if 250 < area < 1000:
            # Find Distance between index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert Volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce Resolution to make it soother
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)
            
            # Check fingers up
            fingers = detector.fingersUp()

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)
            # Hand range 50 ~ 270
            # Volume Range -65 ~ 0    
    
    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

    cv2.putText(img, f'{int(volPer)} %', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cVol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, f'Vol Set: {cVol}', (400, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)


    # Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

volume.SetMasterVolumeLevel(-5.0, None)
cv2.destroyAllWindows()

