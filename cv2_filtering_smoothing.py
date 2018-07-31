import cv2
import numpy as np
# import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # print(frame.shape)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([30, 0, 0])
    upper = np.array([250, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((15, 15), np.float32) / 255
    averaged = cv2.filter2D(res, -1, kernel)

    blur = cv2.GaussianBlur(res, (15, 15), 0)

    median = cv2.medianBlur(res, 15)

    if cv2.waitKey(1) == ord('q'):
        break

    # if _ is False:
    #     break

    cv2.imshow('frame', median)
    # cv2.imshow('res', res)


cv2.destroyAllWindows()
cap.release()
