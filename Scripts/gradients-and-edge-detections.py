import cv2
# import numpy as np
# import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    sobelxy = cv2.Sobel(frame, cv2.CV_64F, 1, 1, ksize=5)

    edge = cv2.Canny(frame, 100, 50)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow('frame', edge)


cv2.destroyAllWindows()
cap.release()
