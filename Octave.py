% Recuperation de l'image dans le vecteur X
X = imread('image_satellite_1.png');

% Recuperation de la composante bleue de l'image dans le vecteur B
B = double(X(:,:,3))/255;

% Affichage de l'image de depart
figure(1);
imshow(X);

% Affichage de la composante bleue de l'image
figure(2);
Xb = zeros(256,3);
imshow(B);
colormap(Xb);
colorbar;
title('composante bleue');
