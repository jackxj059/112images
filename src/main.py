import cv2
from .lbp import LBP

class Foreground:
    def __init__(self):
        self.background=None
    def getForeground(self, frame):
        foreground = frame
        return foreground
    
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
        self.lbp = LBP()
        return
    def lbpFeature(self, img, x, y):
        
        return
    def lbpBackgroundFeature(self, img, x, y):
        return

    def rgbFeature(self, img, x, y):

        return
    def hsvFeature(self, img, x, y):

        return
    
if __name__ =='__main__':
    cap = cv2.VideoCapture(1)

    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()