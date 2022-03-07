import cv2
import time
import numpy as np
import cvzone
from pynput.keyboard import Controller

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

keys = [["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "<"],
        ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"],
        ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "^"]]

shift_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "<"],
              ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
              ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "^"]]




# def drawALL(img, buttonList):
#     for button in buttonList:
#         x,y = button.pos
#         w,h = button.size
#         cvzone.cornerRect(img, (x, y, button.size[0], button.size[1]), 20, rt=0)
#         cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), cv2.FILLED)
#         cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
#                     4, (255,255,255), 4)
#     return img

def drawALL(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x,y = button.pos
        cvzone.cornerRect(imgNew, (x, y, button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x+button.size[0], y+button.size[1]),
                    (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x+40, y+60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    # print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.text = text
        self.size = size
        

buttonList = []
shift_buttonList = []
selected_buttonList = []
finalText = ''
typing_trigger = False
shift_trigger = False
keyboard = Controller()


for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i+50], key))

for i in range(len(shift_keys)):
    for j, key in enumerate(shift_keys[i]):
        shift_buttonList.append(Button([100*j+50, 100*i+50], key))


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    _, img = detector.findHands(img, flipType=False)
    lmList, bboxInfo = detector.findPosition(img)

    if shift_trigger:
        img = drawALL(img, shift_buttonList)
        selected_buttonList = shift_buttonList
    else:
        img = drawALL(img, buttonList)
        selected_buttonList = buttonList

    if lmList:
        for button in selected_buttonList:
            x,y = button.pos
            w,h = button.size

            if x < lmList[8][1] < x+w and y<lmList[8][2]<y+h:
                cv2.rectangle(img, (x-5,y-5), (x+w+5, y+h+5), (175,0,175), cv2.FILLED)
                cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                            4, (255,255,255), 4)
                length, _ = detector.findDistance(lmList[8][1:3], lmList[12][1:3])
                # print(l)

                # when clicked
                if length < 35:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255,255,255), 4)
                    if typing_trigger == False:
                        if button.text == "<":
                            if len(finalText) != 0:
                                finalText = finalText[:-1]
                                typing_trigger = True
                        elif button.text == "^":
                            if shift_trigger:
                                shift_trigger = False
                            else:
                                shift_trigger = True
                            typing_trigger = True
                        else:
                            # pynput
                            keyboard.press(button.text)
                            finalText += button.text
                            typing_trigger = True
                else:
                    typing_trigger = False

    if shift_trigger:
        cv2.rectangle(img, (1050,250), (1135, 335), (0,255,0), cv2.FILLED)
        cv2.putText(img, "^", (1070, 315), cv2.FONT_HERSHEY_PLAIN,
                    4, (255,255,255), 4)      

    cv2.rectangle(img, (50, 350), (700, 450), (175,0,175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN,
                                5, (255,255,255), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.waitKey(1)