import cv2
import numpy as np

print('Librerias leidas...')

#lena
#img = cv2.imread('TAREA/Lena.png')

#ave
img = cv2.imread('TAREA/ave.jpeg')

#actores
#img = cv2.imread('TAREA/actores.jpg')

#girasol
#img = cv2.imread('TAREA/girasol.jpg')

img = cv2.resize(img, (300, 250))

kernel = np.ones((5, 5), np.uint8)
print(kernel)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(img, 200, 200)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel,iterations=1)

cv2.imshow('Imagen original', img)
cv2.imshow('Imagen Gray', imgGray)
cv2.imshow('Imagen Blur', imgBlur)
cv2.imshow('Imagen Canny', imgCanny)
cv2.imshow('Imagen Dilate', imgDialation)
cv2.imshow('Imagen Erode', imgEroded)
cv2.waitKey(0)