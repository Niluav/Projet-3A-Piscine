import numpy as np
import cv2 as cv

#Chargement de l'image
img = cv.imread('image_satellite5.png') #Changer le nom en fonction de l'image à analyser

#Affichage de l'image
cv.imshow('Originale',img)

#Application du 1er seuillage
ret, thresh=cv.threshold(img, 140, 255, cv.THRESH_TOZERO)

cv.imshow('Etape 1',thresh)

#Separation des differents canaux de couleurs
b, g, r = cv.split(thresh) 

#Cree des tableaux identiques a r,g et b rempli de 0
b_const = np.zeros_like(b) 
g_const = np.zeros_like(g)
r_const = np.zeros_like(r)

#Inverse le tableau (0-255 devient 255-0)
b_invert = np.invert(b)

#Re-cree une image BGR avec seulement le bleu
bleu = cv.merge([b,g_const,r_const])

#Re-cree une image en nuance de gris avec les zones contenant du bleu en blanc
bleu_inv = cv.merge([b_invert])

cv.imshow('Etape 2',bleu)

#Applique un flou lege + suppression du bruit
blur = cv.GaussianBlur(img, (3, 3), 0) 

#Passe l'image en nuance de gris
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY) 

#Superposition des 2 images (bleu inverse et nuance de gris)
image_add = cv.add(bleu_inv, gray)

cv.imshow('Etape 3',image_add)

img2 = cv.blur(image_add, (5, 5))

# 2ème seuillage
_, threshold = cv.threshold(image_add, 235, 255, cv.THRESH_BINARY)

# Detecte les contours dans l'image
contours, _= cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
dark_spots = []
pool = 0

# Boucle pour tout les contours trouves
for contour in contours :

    # Calculer le nombre de pixels de chaque contour
    area = cv.contourArea(contour)

    # Conserver seulement les contours les plus grands
    if area > 100:
        pool = pool + 1
        dark_spots.append(contour)

#Affiche le nombre de piscine detectes
print(pool)

#Trace les rectangle a chaques contour trouve
for spot in dark_spots:
    x, y, w, h = cv.boundingRect(spot)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Image finale avec les rectangle sur les piscines
cv.imshow('Etape 4',img)

#Enregistre l'image finale
cv.imwrite("valide.jpg", img)

#Termine le programme dès qu'une touche est appuyee
cv.waitKey(0)
cv.destroyAllWindows()
