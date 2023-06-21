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

class Classfier:
    def __init__(self):
        self.n_neighbor = 5
        self.knn_model = None
        self.bayesian_model = None
        print("loading model....")
        # with open("knnpickle_file_rain3_rgb", 'rb') as file:

        with open("knnpickle_file_rain3", 'rb') as file:
            self.knn_model = pickle.load(file)
            print("loading sucess")
    def knnClassifier(self, feature):
        result = self.knn_model.predict(feature)
        return result

    def bayesianClassifier(self, feature):
        result = None
        return result


class Feature:
    def __init__(self):
        self.lbp = package.LBP()
        self.lbp_background = None

    def updateLbpBackgroun(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.lbp_background = self.lbp.setLBPImageScikit(img)
        
    def lbpFeature(self, img, x=0, y=0):
        if self.lbp_background is None:
            return np.zeros((270,480),dtype=np.uint8 ), np.zeros((270,480),dtype=np.uint8)
        lbp_result = self.lbp.setLBPImageScikit(img)
        subtract_result = cv2.subtract(self.lbp_background, lbp_result)
        # subtract_result[subtract_result<200] = 0
        # print(subtract_result.shape)
        # cv2.imshow()
        return subtract_result, lbp_result
    
    def calMode(self,src,x,y):
        sample = src[y:y+ 10, x:x + 10]
        sample_hist = cv2.calcHist([sample], [0], None, [256], [0, 256])
        counter = Counter(sample_hist.flatten().tolist())
        most_common = counter.most_common(3)
        if len(most_common) ==3 :
            return  most_common[0][1], most_common[1][1], most_common[2][1]
        else:
            return None

    def rgbFeature(self, img, x, y):
        b,g,r = img[x][y]
        return int(b),int(g),int(r)

    def hsvFeature(self, img, x, y):
        return
    
def regTest(fg_contours, fg_mask, frame, feature):
    for cnt in fg_contours:
        contour_size = cv2.contourArea(cnt)
        if contour_size >1000:
            continue
        elif contour_size >700:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.imshow("before", fg_mask[y:h+y, x:x+w])
            cv2.imshow("frame_before" , frame[y:h+y, x:x+w])
            after = np.zeros_like(fg_mask[y:h+y, x:x+w])
            print("contour0")
            print(after.shape)
            for x in range(after.shape[0]):
                for y in range(after.shape[1]):

                    b,g,r = feature.rgbFeature(frame[y:h+y, x:x+w],x, y)
                    res = classfier.knn_model.predict([[b,g,r]])
                    print(res[0])
                    if res[0]:
                        after[x,y] = 0
                    else:
                        after[x,y] = 255

            print("contour1")
            cv2.imshow("after", after)
            cv2.waitKey(0)
            # cv2.destroyWindow("after")
            cv2.destroyAllWindows()
if __name__ =='__main__':
    import time
    import json
    import matplotlib.pyplot as plt
    # classfier = Classfier()
    # path ="video/2023-06-03_12-58.mp4" 
    sample={"object":[],"shadow":[]}
    path ="video/rain" 
    cap = cv2.VideoCapture(path+".mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count)
    feature = Feature()
    # background = Foreground()
    fg = Foreground(mode=1)
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
            if key == ord('q'):
                with open("sample.json","w") as f:
                    print(sample)
                    json.dump(sample, f)
                    f.close()
                return False
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
                # print( len(feature.lbpFeature(frame)))
            end_time=time.time()
            # out.write(draw)
            fg_contours  = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


            draw = np.zeros_like(fg_mask)
            # regTest(fg_contours, fg_mask, frame, feature)
            
            index = 0
            for cnt in fg_contours:
                contour_size = cv2.contourArea(cnt)
                if contour_size >3000:
                    continue

                if contour_size >1000:
                    knn = KnnRemove()                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    img = frame[y:h+y, x:x+w]
                    # mask = np.zeros((img.shape[0],img.shape[1]))
                    mask = fg_mask[y:h+y, x:x+w]
                    cv2.drawContours(mask, cnt, -1, 255, 3)
                    cv2.imshow("img" ,img )
                    cv2.imshow('mask' ,mask)

                    knn.sampling(img, mask)
                    knn.train()
                    result = knn.predict(img)
                    draw[y:h+y, x:x+w] = result
                index += 1
            cv2.imshow('draw',draw)
            # try:
            #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            #     # draw = cv2.morphologyEx(cv2.bitwise_and(sub_threshold,fg_mask ), cv2.MORPH_CLOSE, kernel)
            # except:
            #     pass
            
            if developer:
                # cv2.imshow('frame',cv2.bitwise_and(frame, frame, mask=fg_mask))
                cv2.imshow('frame',frame)
                # cv2.imshow('sub_thrshold', sub_threshold)
                cv2.imshow('fg_mask', fg_mask)
                if not keyboardTool():
                    break
            # print(round((fc1/frame_count)*100))
            fc += 1
            fc1 += 1
    except Exception as e:
        with open("sample_rain2.json","w") as f:
            json.dump(sample, f)
            f.close()
        cap.release()
        out.release()
        print(e)
        print("end")