import cv2
import time
import numpy as np
import cvzone
from pynput.keyboard import Controller
from PIL import ImageFont, ImageDraw, Image
from hangul_utils import split_syllable_char, split_syllables, join_jamos

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["ㅂ", "ㅈ", "ㄷ", "ㄱ", "ㅅ", "ㅛ", "ㅕ", "ㅑ", "ㅐ", "ㅔ", "<"],
        ["ㅁ", "ㄴ", "ㅇ", "ㄹ", "ㅎ", "ㅗ", "ㅓ", "ㅏ", "ㅣ", ";"],
        ["ㅋ", "ㅌ", "ㅊ", "ㅍ", "ㅠ", "ㅜ", "ㅡ", ",", ".", "/", "^"]]

shift_keys = [["ㅃ", "ㅉ", "ㄸ", "ㄲ", "ㅆ", "ㅛ", "ㅕ", "ㅑ", "ㅒ", "ㅖ", "<"],
             ["ㅁ", "ㄴ", "ㅇ", "ㄹ", "ㅎ", "ㅗ", "ㅓ", "ㅏ", "ㅣ", ";"],
             ["ㅋ", "ㅌ", "ㅊ", "ㅍ", "ㅠ", "ㅜ", "ㅡ", ",", ".", "/", "^"]]

fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
font = ImageFont.truetype(fontpath, 35)
font_big = ImageFont.truetype(fontpath, 70)


def drawALL(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x,y = button.pos
        cvzone.cornerRect(imgNew, (x, y, button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x+button.size[0], y+button.size[1]),
                    (255, 0, 255), cv2.FILLED)
        # 한글 적용        
        img_pil = Image.fromarray(imgNew)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x+40, y+30), f'{button.text}', font=font, fill=(255,255,255,0))
        imgNew = np.array(img_pil)

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
combinedText = ''
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
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x+40, y+30), f'{button.text}', font=font, fill=(255,255,255,0))
                img = np.array(img_pil)
                length, _ = detector.findDistance(lmList[8][1:3], lmList[12][1:3])

                # when clicked
                if length < 35:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), cv2.FILLED)
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x+40, y+30), f'{button.text}', font=font, fill=(255,255,255,0))
                    img = np.array(img_pil)
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
                            # keyboard.press(button.text)
                            finalText += button.text
                            typing_trigger = True
                else:
                    typing_trigger = False         

    if shift_trigger:
        cv2.rectangle(img, (1050,250), (1135, 335), (0,255,0), cv2.FILLED)
        cv2.putText(img, "^", (1070, 315), cv2.FONT_HERSHEY_PLAIN,
                    4, (255,255,255), 4)                  

    try:
        combinedText = join_jamos(finalText)
    except:
        pass

    cv2.rectangle(img, (50, 350), (700, 450), (175,0,175), cv2.FILLED)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((60, 350), f'{combinedText}', font=font_big, fill=(255,255,255,0))
    img = np.array(img_pil)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.waitKey(1)