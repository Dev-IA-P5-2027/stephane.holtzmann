import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

cv.waitKey(0)