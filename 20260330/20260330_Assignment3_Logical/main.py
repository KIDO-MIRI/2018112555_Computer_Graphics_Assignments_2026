import cv2
import numpy as np

img1 = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Images not found")
else:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    _, bin1 = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

    bitwise_and = cv2.bitwise_and(bin1, bin2)
    bitwise_or = cv2.bitwise_or(bin1, bin2)
    bitwise_not = cv2.bitwise_not(bin1)

    cv2.imshow('Binary 1', bin1)
    cv2.imshow('Binary 2', bin2)
    cv2.imshow('AND', bitwise_and)
    cv2.imshow('OR', bitwise_or)
    cv2.imshow('NOT', bitwise_not)

    cv2.waitKey(0)
    cv2.destroyAllWindows()