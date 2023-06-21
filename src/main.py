import sys
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
        self.object = object
    def setShadow(self, shadow):
        self.shadow = shadow
    def train(self):
        self.model = KNeighborsClassifier()
        derc = self.object.tolist()
        derc.extend(self.shadow)
 
        label = [0]*len(self.object)
        label.extend([1]*len(self.shadow))
        self.model.fit(derc, label)
    def sampling(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
        harris_mask = np.zeros_like(mask, dtype=np.uint8) 
        harris_mask[dst > 0.01 * dst.max()] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        harris_mask = cv2.dilate(harris_mask, kernel, iterations=2)   
        knn.setObject(img[harris_mask==255])

    def predict(self, img):
        result = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        height, width = result.shape
        for y in range(height):
            for x in range(width):
                b, g, r = img[y, x]
                if self.model.predict([[b, g, r]]) == 0:
                    result[y, x] = 255
                else:
                    result[y, x] = 0
        # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  
        return result

def on_mouse(event, x, y, flags):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 2)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img1 = img[min_y:min_y + height, min_x:min_x + width]
        # cv2.imwrite('imgs/Area1.jpg', cut_img1)

if __name__ =='__main__':
    import time

    path =sys.argv[1]
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    fg = Foreground(mode=1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    bg = fg.getBackground()
    draw = None
    ret = True
    fc = 0
    fc1 = 0
    developer = True
    pre_frame_result = None
    remove_high_light = False 
    point1 = None
    point2 = None
    cut_img1 = None
    draw_img = None


    shadow_feature = None
    shadow_data = []
    def on_mouse(event, x, y, flags, param):
        global draw_img, point1, point2, cut_img1, shadow_feature
        img2 = draw_img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(img2, point1, 10, (0, 255, 0), 2)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(img2, point1, point2, (0, 0, 255), 2)
            cv2.imshow('image', img2)
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            shadow_feature = draw_img[min_y:min_y + height, min_x:min_x + width]

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
        ret, draw_img = cap.read()
        draw_img = cv2.resize(draw_img, (480,270))
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', draw_img)
        cv2.waitKey(0)

        height, width, channels = shadow_feature.shape

        for y in range(height):
            for x in range(width):
                b, g, r = draw_img[y, x]
                shadow_data.append([b, g, r])
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
                if contour_size >50000:
                    continue

                if contour_size >400:

                    knn = KnnRemove()                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    img = frame[y:h+y, x:x+w]
                    mask = fg_mask[y:h+y, x:x+w]
                    knn.sampling(img)
                    knn.setShadow(shadow_data)
                    knn.train()
                    res = knn.predict(img)

                    draw[y:h+y, x:x+w] = res
 
            if developer:
                cv2.imshow('draw',draw)
                cv2.imshow('frame', frame)
                cv2.imshow('fg_mask', fg_mask)
                if not keyboardTool():
                    break
            # print(round((fc1/frame_count)*100))
            fc += 1
            fc1 += 1
    except:
        cap.release()
        print("end")