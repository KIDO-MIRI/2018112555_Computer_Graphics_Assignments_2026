import cv2
import numpy as np

img1 = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Images not found")
else:
    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5) #scale image
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5) #scale image
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    add_res = cv2.add(img1, img2)
    sub_res = cv2.subtract(img1, img2)
    mul_res = cv2.multiply(img1, np.array([1.5])).astype(np.uint8)
    div_res = cv2.divide(img1, np.array([2.0])).astype(np.uint8)

    cv2.imshow('Original 1', img1)
    cv2.imshow('Addition', add_res)
    cv2.imshow('Subtraction', sub_res)
    cv2.imshow('Multiplication', mul_res)
    cv2.imshow('Division', div_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()