import cv2
# import numpy as np
# import matplotlib.pyplot as plt

video = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    _, frame = video.read()
    fgmask = fgbg.apply(frame)

    # cv2.imshow('frame', frame)
    cv2.imshow('mask', fgmask)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
