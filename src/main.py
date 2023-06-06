import cv2
import package
import numpy as np
class Foreground:
    def __init__(self):
        self.background=None
        # self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=50)
        self.backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    def getForeground(self, frame):
        fg_mask = self.backSub.apply(frame)
        
        # foreground = frame
        return fg_mask
    def getBackground(self):

        return self.backSub.getBackgroundImage()
class Classfier:
    def __init__(self):
        self.n_neighbor = 5
        self.knn_model = None
        self.bayesian_model = None

    def knnClassifier(self,feature):
        result = None 
        return result
    
    def bayesianClassifier(self,feature):
        result = None 
        return result


class Feature:
    def __init__(self):
        self.lbp = package.LBP()
        self.lbp_background = None
        # return

    def lbpFeature(self, img, x=0, y=0):
        lbp_result = self.lbp.setLBPImageScikit(img)
        
        # self.lbp = 
        return lbp_result
    def updateLbpBackgroun(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.lbp_background = self.lbp.setLBPImageScikit(img)
        cv2.imshow('feature', self.lbp_background)
        
    def lbpBackgroundFeature(self, img, x=0, y=0):
        if self.lbp_background is None:
            return np.zeros_like(img)
        lbp_result = self.lbp.setLBPImageScikit(img)
        subtract_result = cv2.subtract(self.lbp_background, lbp_result)
        subtract_result[subtract_result<200] = 0
        # print(subtract_result.shape)
        cv2.imshow()
        return subtract_result

    def rgbFeature(self, img, x, y):

        return
    def hsvFeature(self, img, x, y):

        return
def keyboardTool():
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('s'):
            stop =True
            while stop:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    stop = False
                    return True
        return True
if __name__ =='__main__':
    import time
    import matplotlib.pyplot as plt
    # path ="video/2023-06-03_12-58.mp4" 
    path ="video/441k901_2022-05-24_23-00.mp4" 
    cap = cv2.VideoCapture(path)
    feature = Feature()
    background = Foreground()
    draw = None
    ret = True
    fc = 0
    pre_frame_result = None
    remove_high_light = True 
    while(ret):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (480,270))
        draw = np.zeros((480,270))
        start_time=time.time()
        if not frame is None:
            if remove_high_light:
                frame[frame> 180 ]=0

            fg_mask = background.getForeground(frame)
            sub_result = feature.lbpBackgroundFeature(frame)
            _, sub_threshold = cv2.threshold(sub_result,210,255,cv2.THRESH_BINARY)
            # print(sub_result.shape, sub_threshold.shape)
        if fc % 180 == 0:
            feature.updateLbpBackgroun(background.getBackground())
            fc =0 
    
        end_time=time.time()

        # draw[sub_threshold !=0]=255
        # print(1/(end_time-start_time))


        cv2.imshow('frame', frame)
        cv2.imshow('sub_threshold', sub_threshold)
        cv2.imshow('sub_result', sub_result)
        cv2.imshow('fg_mask', fg_mask)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            draw = cv2.morphologyEx(cv2.bitwise_and(sub_threshold,fg_mask ), cv2.MORPH_CLOSE, kernel)
            cv2.imshow('draw',draw)
        except:
            pass

        if not keyboardTool():
            break
        # if key == ord('w'):
        #     while 1:
        #        key =  cv2.waitKey(1) & 0xFF
        #        if key == ord('w'):
        #            break 
        fc += 1
    cap.release()
    cv2.destroyAllWindows()