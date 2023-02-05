import numpy as np
import cv2 as cv

#Chargement de l'image
img = cv.imread('image_satellite4.png') 

#Affichage de l'image
cv.imshow('Originale',img)

#Application du 1er seuillage
ret, thresh=cv.threshold(img, 140, 255, cv.THRESH_TOZERO)

cv.imshow('Etape 1',thresh)

#Séparation des différents canaux de couleurs
b, g, r = cv.split(thresh) 

#Crée des tableaux identiques a r,g et b rempli de 0
b_const = np.zeros_like(b) 
g_const = np.zeros_like(g)
r_const = np.zeros_like(r)

#Inverse le tableau (0-255 devient 255-0)
b_invert = np.invert(b)

#Re-crée une image BGR avec seulement le bleu
bleu = cv.merge([b,g_const,r_const])

#Re-crée une image en nuance de gris avec les zones contenant du bleus en blanc
bleu_inv = cv.merge([b_invert])

cv.imshow('Etape 2',bleu)

#Applique un flou légé + suppression du bruit
blur = cv.GaussianBlur(img, (3, 3), 0) 

#Passe l'image en nuance de gris
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY) 


cv.imshow('Etape 3b',gray)

#Superposition des 2 images (bleu inversé et contours)
image_add = cv.add(bleu_inv, gray)

#Utilisation de l'algorithme de Sobel(permet de récuperer les contours)
grad_x = cv.Sobel(image_add, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(image_add, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

inverted_image = cv.bitwise_not(grad)
img2 = cv.blur(inverted_image, (5, 5))

cv.imshow('Etape 4', inverted_image)

# Converti l'image seulement en noir et blanc
_, threshold = cv.threshold(grad, 8.5, 255, cv.THRESH_TOZERO)
# Detecte les contours dans l'image
contours, _= cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
dark_spots = []
pool = 0
# Boucle pour tout les contours trouvés
for contour in contours :

    # Calculer le nombre de pixels de chaque contour
    area = cv.contourArea(contour)

    # Conserver seulement les contours les plus grands
    if  area > 700:
        pool = pool +1
        dark_spots.append(contour)

#Affiche le nombre de piscine détectées
print(pool)

#Trace les rectangle a chaques contour trouvé
for spot in dark_spots:
    x, y, w, h = cv.boundingRect(spot)
    if 300 < w*h < 3400:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Image finale avec les rectangle sur les piscines
cv.imshow('Etape 5',img)

#Enregistre l'image finale
cv.imwrite("outputimagetes.jpg", img)

#Termine le programme dès qu'une touche est appuyée
cv.waitKey(0)
cv.destroyAllWindows()
