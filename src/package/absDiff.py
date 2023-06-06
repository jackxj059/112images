import cv2
import sys
import time

cap = cv2.VideoCapture(sys.argv[1])

ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = cap.read()

    if not ret:
        break

    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_frame, curr_frame)
    print(diff)

    _, contours, hierarchy = cv2.findContours(
        diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 800:
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.imwrite(f"../fgImage/shadow/shadow_{time.time()}.jpg", curr_frame[y:y + h, x:x + w])
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # cv2.imshow('Frame', curr_frame)
    cv2.imshow('Difference', diff)

    prev_frame = curr_frame

    if cv2.waitKey(0) == 27:
        break

cap.release()
cv2.destroyAllWindows()
