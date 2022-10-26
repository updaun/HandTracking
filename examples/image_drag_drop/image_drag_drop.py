import cv2
import cvzone

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from modules.HandTrackingModule_cvzone import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=1, detectionCon=0.65)

class DragImg():
    def __init__(self, path, posOrigin, imgType):
        self.path = path
        self.posOrigin = posOrigin
        self.imgType = imgType

        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)

        self.size = self.img.shape[:2]
    
    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size
        # Check if in region
        if ox<cursor[0]<ox+w and oy<cursor[1]<oy+h:
            # Inside Image
            self.posOrigin = cursor[0]-w//2, cursor[1]-h//2
            

# img1 = cv2.imread('examples/image_drag_drop/Images/ImagesJPG/1.jpg')
# img1 = cv2.imread('examples/image_drag_drop/Images/ImagesPNG/1.png', cv2.IMREAD_UNCHANGED)
# ox, oy = 500, 200

path = "examples/image_drag_drop/Images/ImagesMix"
myList = os.listdir(path)
# print(myList)

listImg = []
for x, pathImg in enumerate(myList):
    if 'png' in pathImg:
        imgType = 'png'
    else:
        imgType = 'jpg'
    listImg.append(DragImg(f'{path}/{pathImg}', [50+x*300, 50], imgType))


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']

        # Check if clicked
        length, info, img = detector.findDistance(lmList[4][1:3], lmList[12][1:3], img)
        print("distance : ", length)
        if length < 80:
            cursor = lmList[8]
            for imgObject in listImg:
                imgObject.update(cursor)

    try:
        for imgObject in listImg:
            h, w = imgObject.size
            ox, oy = imgObject.posOrigin
            if imgObject.imgType == "png":
                # Draw for PNG image
                img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
            else:
                # Draw for JPG image
                img[oy:oy+h, ox:ox+w] = imgObject.img
    except:
        pass

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break
    cv2.waitKey(1)
    