import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lecture de l'image source
image = cv2.imread("image_satellite_2.png")

# Conversion de l'image en RVB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Transformation de l'image en un tableau 2D de pixels et de trois couleurs (RVB)
pixel_values = image.reshape((-1, 3))
# Conversion en flotant
pixel_values = np.float32(pixel_values)
# Affichage des valeurs du tableau de pixels obtenu
print(pixel_values.shape)
(2073600, 3)

# Definition du critere d'arret des iterations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Nombre de clusters (k)
k = 20
# Definition des clusters, des labels et de leurs points centre
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Conversion de nouveau en valeurs de 8 bits
centers = np.uint8(centers)
# Conversion du tableau de labels en tableau d'une seule dimension
labels = labels.flatten()

# Conversion de tous les pixels en la couleur des centroids
segmented_image = centers[labels.flatten()]

# Transformation de l'image afin de retrouver les dimensions de l'image d'origine
segmented_image = segmented_image.reshape(image.shape)
# Affichage de l'image obtenue
plt.figure(1)
plt.imshow(segmented_image)


# Masquage de certains clusters
# Creation d'une copie de l'image obtenue
masked_image = np.copy(image)
# Conversion de l'image en vecteur de valeurs de pixels
masked_image = masked_image.reshape((-1, 3))
# Selection des labels des clusters que l'on souhaite masquer
cluster1 = 1
cluster2 = 6
# Masquage de clusters selectionnes (ici 1 et 6) (Les pixels deviennent noir)
masked_image[labels == cluster1] = [0, 0, 0]
masked_image[labels == cluster2] = [0, 0, 0]
# Conversion au format d'origine
masked_image = masked_image.reshape(image.shape)
# Affichage de l'image
plt.figure(2)
plt.imshow(masked_image)
plt.show()
