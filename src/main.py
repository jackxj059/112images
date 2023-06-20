import sys

import cv2
import pickle
import random
import package
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from numba import  jit
class Foreground:
    def __init__(self, mode):
        self.background = None

        if mode == 1:

            self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
        elif mode == 2:

            self.backSub = cv2.createBackgroundSubtractorKNN(detectShadows = True)
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
    def setObject(self, src):
        self.object = src
    def setShadow(self, shadow):
        self.shadow = shadow
    def train(self):
        return
    def predict(self, img):
        result = img
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
    classfier = Classfier()
    # path ="video/2023-06-03_12-58.mp4" 
    sample={"object":[],"shadow":[]}
    path ="video/rain" 
    cap = cv2.VideoCapture(path+".mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count)
    feature = Feature()
    # background = Foreground()
    fg = Foreground(mode=2)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+'-output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (960,  540))
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
            
            if key == ord('w'):
                shadow = np.zeros_like(fg_mask)
                obj = np.zeros_like(fg_mask)
                shadow[fg_mask==127]=255
                obj[fg_mask==255]=255
                cv2.imshow("shadow", shadow)
                cv2.imshow("obj", obj)
                stop =True

                while stop:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('w'):
                        stop = False
                        return True
                    if key == ord('y'):
                        shadow_contours  = cv2.findContours(shadow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                        obj_contours  = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                        for shadow_cnt in shadow_contours:
                            x, y, w, h = cv2.boundingRect(shadow_cnt)
                            src = frame[y:y+h,x:x+w]
                            for i in range(0,3):
                                px = random.randrange(0,w,10)
                                py = random.randrange(0,h,10)
                                b,g,r = frame[px][py]
                                sample["shadow"].append([int(b),int(g),int(r)])
                        for obj_cnt in obj_contours:
                            x, y, w, h = cv2.boundingRect(obj_cnt)
                            src = frame[y:y+h,x:x+w]
                            for i in range(0,3):
                                px = random.randrange(0,w,10)
                                py = random.randrange(0,h,10)
                                b,g,r = frame[px][py]
                                sample["object"].append([int(b),int(g),int(r)])
                        stop = False
                        return True
                    if key == ord('x'):
                        # fig = plt.figure()
                        # axmode = fig.gca(projection='3d')
                        shadow_contours  = cv2.findContours(shadow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                        obj_contours  = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                        for shadow_cnt in shadow_contours:
                            x, y, w, h = cv2.boundingRect(shadow_cnt)
                            src = lbp_result[y:y+h,x:x+w]
                            src2 = sub_result[y:y+h,x:x+w]
                            for i in range(0,3):
                                px = random.randrange(0,w,10)
                                py = random.randrange(0,h,10)
                                mode = feature.calMode(src,px,py)
                                mode2 = feature.calMode(src2,px,py)
                                if not mode is None and not mode2 is None :
                                    # axmode.scatter(mode[0], mode[1], mode[2], c="b", cmap='Reds', marker="s")
                                    sample["shadow"].append([mode[0], mode[1], mode[2], mode2[0], mode2[1], mode2[2]])
                        for obj_cnt in obj_contours:
                            x, y, w, h = cv2.boundingRect(obj_cnt)
                            src = lbp_result[y:y+h,x:x+w]
                            src2 = sub_result[y:y+h,x:x+w]
                            for i in range(0, 3):
                                px = random.randrange(0,w,10)
                                py = random.randrange(0,h,10)
                                mode = feature.calMode(src,px,py)
                                mode2 = feature.calMode(src2,px,py)
                                if not mode is None and not mode2 is None :
                                    # axmode.scatter(mode[0], mode[1], mode[2], c="c", cmap='Reds', marker="o")
                                    sample["object"].append([mode[0], mode[1], mode[2], mode2[0], mode2[1], mode2[2]])
                        # axmode.legend()
                        # plt.show()
                        stop = False
                        return True
            return True
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
                sub_result,lbp_result = feature.lbpFeature(frame)

                # _, sub_threshold = cv2.threshold(sub_result,220,255,cv2.THRESH_BINARY)
                # print(sub_result.shape, sub_threshold.shape)
                 

            draw[sub_result >250] = 255
            if fc % 180 == 0:
                feature.updateLbpBackgroun(fg.getBackground())
                fc =0 
        
            end_time=time.time()
            
            # print(1/(end_time-start_time))
            # lbp_result[fg_mask == 0] = 0
            # sub_result[fg_mask == 0] = 0
            lbp_result = cv2.cvtColor(lbp_result, cv2.COLOR_GRAY2BGR)
            sub_result = cv2.cvtColor(sub_result, cv2.COLOR_GRAY2BGR) 

            cv2.putText(lbp_result, "lbp_result", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(sub_result, "sub_result", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)
            # draw[0:270,480:960] = lbp_result
            # draw[270:540,0:480] = sub_result
            # draw[0:270,0:480]   = frame

            # out.write(draw)
            fg_contours  = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

            # regTest(fg_contours, fg_mask, frame, feature)

            for cnt in fg_contours:
                contour_size = cv2.contourArea(cnt)
                if contour_size >1000:
                    continue

                if contour_size >700:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.imshow("before", fg_mask[y:h+y, x:x+w])
                    cv2.imshow("frame" , frame[y:h+y, x:x+w])
                    after = np.zeros_like(fg_mask[y:h+y, x:x+w])
                    print("contour0")
                    for x in range(after.shape[0]):
                        for y in range(after.shape[1]):

                            mode = feature.calMode(sub_result[y:h+y, x:x+w],x, y)
                            mode2 = feature.calMode(lbp_result[y:h+y, x:x+w],x, y)

                            if not mode2 is None and not mode is None:
                                res = classfier.knn_model.predict([[mode[0], mode[1], mode[2], mode2[0], mode2[1], mode2[2]]])
                            
                                if res[0]:
                                    after[x,y] = 0
                                else:
                                    after[x,y] = 255

                    print("contour1")
                    cv2.imshow("after", after)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                # draw = cv2.morphologyEx(cv2.bitwise_and(sub_threshold,fg_mask ), cv2.MORPH_CLOSE, kernel)
                draw = cv2.dilate(draw, kernel, iterations = 1)
                cv2.imshow('draw',draw)
            except:
                pass
            
            if developer:
                cv2.imshow('frame', frame)
                # cv2.imshow('sub_threshold', sub_threshold)
                cv2.imshow('fg_mask', fg_mask)
                cv2.imshow('lbp_result', lbp_result)
                cv2.imshow('sub_result', sub_result)
                if not keyboardTool():
                    break
            print(round((fc1/frame_count)*100))
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