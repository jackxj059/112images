import cv2
import numpy as np
import argparse
import time
import MultiThreshold

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str,
                        help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str,
                        help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()

    # [create]
    # create Background Subtractor objects
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=20)
        # backSub = cv2.bgsegm.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    # back_sub = cv2.bgsegm.createBackgroundSubtractorMOG()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # [create]

    # [capture]
    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)
    # [capture]

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        # [apply]
        # update the background model
        fg_mask = backSub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # bg_mask = backSub.getBackgroundImage()
        # th = cv2.threshold(fg_mask.copy(), 244, 250, cv2.THRESH_BINARY)[1]
        #
        # dialated = cv2.dilate(th, kernel, iterations=2)
        # [apply]

        # [display_frame_number]
        # get the frame number and write it on the current frame
        # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        # [display_frame_number]

        object = np.zeros_like(fg_mask, dtype=np.uint8)
        object[fg_mask == 255] = 255

        shadow = np.zeros_like(fg_mask, dtype=np.uint8)
        shadow[fg_mask == 127] = 255

        _, contours, hierarchy = cv2.findContours(
            object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list = []
        for c in contours:
            if cv2.contourArea(c) > 800:
                # contours_list.append(c)
                (x, y, w, h) = cv2.boundingRect(c)
                # cv2.imwrite(f"../fgImage/shadow/shadow_{time.time()}.jpg", frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # [show]
        # show the current frame and the fg masks
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fg_mask)
        cv2.imshow('Shadow', object)

        # cv2.drawContours(frame, contours_list, -1, (0, 255, 0), 3)
        # cv2.imshow('Contour', frame)

        # cv2.imshow('BG Mask', bg_mask)
        # [show]

        keyboard = cv2.waitKey(30)

        # press ESC or q to exit
        if keyboard == 113 or keyboard == 27:
            break
        # press w to freeze the frame
        elif keyboard == 119:
            while True:
                keyboard = cv2.waitKey(0)
                if keyboard == 119:
                    break

    capture.release()
    cv2.destroyAllWindows()
