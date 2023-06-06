import sys

import cv2
import package
import numpy as np


class Foreground:
    def __init__(self, mode):
        self.background = None

        if mode == 1:
            self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        elif mode == 2:
            self.backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
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


class Classfier:
    def __init__(self):
        self.n_neighbor = 5
        self.knn_model = None
        self.bayesian_model = None

    def knnClassifier(self, feature):
        result = None
        return result

    def bayesianClassifier(self, feature):
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
        subtract_result[subtract_result < 200] = 0
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
        stop = True
        while stop:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                stop = False
                return True
    return True


if __name__ == '__main__':
    capture = cv2.VideoCapture(sys.argv[1])
    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)

    fg = Foreground(mode=2)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fg_mask = fg.getForeground(frame)
        bg = fg.getBackground()

        cv2.imshow('Frame', frame)
        cv2.imshow('fg', fg_mask)
        cv2.imshow('bg', bg)
        keyboardTool()

    capture.release()
    cv2.destroyAllWindows()