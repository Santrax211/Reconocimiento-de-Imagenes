import cv2
import numpy as np

print('Librerias leidas...')

# Carga la imagen que se quiere procesar
# Otras imágenes están comentadas, se puede descomentar para usarlas.
#img = cv2.imread('TAREA/Lena.png') # Lena
#img = cv2.imread('TAREA/ave.jpeg')  # Ave
#img = cv2.imread('TAREA/actores.jpg') # Actores
#img = cv2.imread('TAREA/girasol.jpg') # Girasol
#img = cv2.imread('TAREA/will-smith.jpg')
#img = cv2.imread('TAREA/leonardo.jpg')
#img = cv2.imread('TAREA/alexandra.jpg')
img = cv2.imread('TAREA/balotelli.jpg')

# Redimensiona la imagen a 300 píxeles de ancho y 250 píxeles de alto
img = cv2.resize(img, (300, 250))

# Crea un kernel de 5x5 lleno de unos, que se usará para dilatar y erosionar la imagen
kernel = np.ones((5, 5), np.uint8)
print(kernel)

# Convierte la imagen a escala de grises
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica un filtro de desenfoque gaussiano para suavizar la imagen
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

# Aplica el algoritmo Canny para detectar bordes en la imagen original
imgCanny = cv2.Canny(img, 200, 200)

# Realiza una dilatación de los bordes detectados, aumentando el grosor de los mismos
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

# Realiza una erosión sobre la imagen dilatada, reduciendo el grosor de los bordes
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

alpha = 1.5  # Contraste
beta = 50    # Brillo
imgBright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#Reliza una inversión negativa de la imagen inicial
imgInverse = cv2.bitwise_not(img)

# Detecta bordes en la dirección X
imgSobelX = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)

# Detecta bordes en la dirección Y
imgSobelY = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)  

# Transforma la imagen en una imagen binaria según un umbral, donde los píxeles se clasifican como blancos o negros.
_, imgThresh = cv2.threshold(imgGray, 120, 255, cv2.THRESH_BINARY)

# Muestra las distintas versiones de la imagen procesada en ventanas separadas
cv2.imshow('Imagen original', img)  # Muestra la imagen original
cv2.imshow('Imagen Gray', imgGray)  # Muestra la imagen en escala de grises
cv2.imshow('Imagen Blur', imgBlur)  # Muestra la imagen desenfocada
cv2.imshow('Imagen Canny', imgCanny)  # Muestra la imagen con los bordes detectados
cv2.imshow('Imagen Dilate', imgDialation)  # Muestra la imagen con bordes dilatados
cv2.imshow('Imagen Erode', imgEroded)  # Muestra la imagen con bordes erosionados
cv2.imshow("Brightened Image", imgBright) #Muestra la imagen con bordes dilatados con una mayor luminosidad
cv2.imshow("Inverted Image", imgInverse) #Muestra la imagen invertida negativa
cv2.imshow("Sobel X", imgSobelX) # Muestra la imagen con los bordes en la dirección X
cv2.imshow("Sobel Y", imgSobelY) #  Muestra la imagen con los bordes en la dirección Y
cv2.imshow("Threshold Image", imgThresh) # Muestra la imagen binaria

# Espera indefinidamente a que el usuario presione una tecla para cerrar las ventanas
cv2.waitKey(0)
