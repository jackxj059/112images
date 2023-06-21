import sys

import cv2
import pickle
import random
import package
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from numba import  jit
class Foreground:
    def __init__(self, mode):
        self.background = None

        if mode == 1:
            self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
        elif mode == 2:
            self.backSub = cv2.createBackgroundSubtractorKNN(detectShadows = False)
            # print(self.backSub.getShadowValue())
        elif mode == 3:
            self.backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    def getForeground(self, frame):
        _fg_mask = self.backSub.apply(frame)
        self.background = self.backSub.getBackgroundImage()
        _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _fg_mask = cv2.morphologyEx(_fg_mask, cv2.MORPH_OPEN, _kernel)
        return _fg_mask

    def getBackground(self):
        return self.background
class KnnRemove:
    def __init__(self):
        self.object = None
        self.shadow = None
        self.model  = None
    def setObject(self, object):
        self.object = object #紀錄物件特徵
    def setShadow(self, shadow):
        self.shadow = shadow #紀錄陰影特徵

    def sampling(self, img, fg_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask = cv2.erode(fg_mask, kernel, iterations = 1) #進行侵蝕找到只屬於物件的區域
        shadow_mask = cv2.dilate(fg_mask, kernel, iterations = 1)#進行膨脹找到只屬於陰影的區域
        self.setObject(img[object_mask == 255])
        self.setShadow(img[shadow_mask == 0])
        # cv2.imshow('obj' ,cv2.bitwise_and(img, img, mask=object_mask))
        # cv2.imshow('shw' ,cv2.bitwise_and(img, img, mask=~shadow_mask))
        # cv2.imshow('obj2' ,object_mask)
        # cv2.imshow('shw2' ,shadow_mask)

    def train(self):
        self.model = KNeighborsClassifier()
        derc = self.object.tolist()
        derc.extend(self.shadow.tolist())
        label = [0]*len(self.object)
        label.extend([1]*len(self.shadow))
        self.model.fit(derc, label) # 訓練knn 模型
    
    def predict(self, img):
        result = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        height, width = result.shape
        for y in range(height):
            for x in range(width):
                b, g, r = img[y, x]
                if self.model.predict([[b, g, r]]) == 0:#針對每個pixel進行預測
                    result[y, x] = 255
                else:
                    result[y, x] = 0
        return result
if __name__ =='__main__':
    import time
    path = sys.argv[1]
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fg = Foreground(mode=2)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+'-output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (480,270))
    bg = fg.getBackground()
    draw = None
    ret = True
    fc = 0
    fc1 = 0
    developer = True
    pre_frame_result = None
    remove_high_light = False 
    
    def keyboardTool():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            stop =True
            while stop:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    stop = False
                    return True
        return True
#---------------------------------------------------------------------------------------------
    try:
        while(ret):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (480,270))
            draw = np.zeros_like(frame)
            start_time=time.time()
            if not frame is None:
                if remove_high_light:
                    frame[frame > 180]=0
                fg_mask = fg.getForeground(frame)
            end_time=time.time()
            fg_contours  = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


            draw = np.zeros_like(fg_mask)
            
            for cnt in fg_contours:
                contour_size = cv2.contourArea(cnt)
                if contour_size >4000:
                    continue

                if contour_size >500:
                    knn = KnnRemove()                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    img = frame[y:h+y, x:x+w]
                    mask = fg_mask[y:h+y, x:x+w]
                    cv2.drawContours(mask, cnt, -1, 255, 3)
                    knn.sampling(img, mask)
                    knn.train()
                    result = knn.predict(img)
                    draw[y:h+y, x:x+w] = result
            cv2.imshow('draw',draw)
            if developer:
                cv2.imshow('frame',frame)
                cv2.imshow('fg_mask', fg_mask)
                if not keyboardTool():
                    break
            fc += 1
            fc1 += 1
    except Exception as e:
        cap.release()
        out.release()
        print(e)
        print("end")