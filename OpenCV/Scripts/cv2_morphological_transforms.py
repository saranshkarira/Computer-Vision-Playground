import cv2
import numpy as np
# import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([30, 0, 0])
    upper = np.array([250, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((2, 2), np.uint8)

    erosion = cv2.erode(res, kernel, iterations=1)
    dilation = cv2.dilate(res, kernel, iterations=1)

    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    combo = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow('frame', combo)


cv2.destroyAllWindows()
cap.release()
