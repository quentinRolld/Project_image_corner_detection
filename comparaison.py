
### Fonctions pour la comparaison

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from skimage.io import imread,imshow
from skimage.color import rgb2gray
from skimage.transform import rotate
from matplotlib import patches
import cv2

from fast import fast_1
from harris import detection_harris_gauss_optimized



def comp_soustraction( img_path ):

    img = imread(img_path)

    img_FAST, _ = fast_1(img_path, 0.05, 12, 0)
    img_Harris_Gauss, _ = detection_harris_gauss_optimized(img_path, 0, 3, 0.04, 2)

    # On convertit les images en niveaux de gris
    img_FAST_gray = rgb2gray(img_FAST)
    img_Harris_Gauss_gray = rgb2gray(img_Harris_Gauss)

    # On crée des masques pour les coins détectés
    mask_FAST = np.zeros_like(img_FAST_gray)
    mask_Harris_Gauss = np.zeros_like(img_Harris_Gauss_gray)

    # On récupère les coordonnées des coins détectés
    corners_FAST = np.argwhere(np.all(img_FAST == [255, 0, 0], axis=-1))
    corners_Harris_Gauss = np.argwhere(np.all(img_Harris_Gauss == [255, 0, 0], axis=-1))

    # On met à 1 les pixels correspondant aux coins détectés
    for corner in corners_FAST:
        mask_FAST[corner[0], corner[1]] = 1

    for corner in corners_Harris_Gauss:
        mask_Harris_Gauss[corner[0], corner[1]] = 1

    # On calcule le XOR des deux masques pour avoir les points d'intérêts uniques à chaque méthode
    mask_XOR = np.logical_xor(mask_FAST, mask_Harris_Gauss)

    points = np.argwhere(mask_XOR == 1)
    print("Nombre de points d'intérêts uniques à chaque méthode :", len(points))

    # On crée une image avec les points d'intérêts
    img_points = np.zeros_like(img)
    img_points[mask_XOR, 0] = 255  # points en rouge pour FAST
    img_points[mask_XOR, 1] = 0
    img_points[mask_XOR, 2] = 0
    img_points[np.logical_and(~mask_XOR, mask_Harris_Gauss), 0] = 0  # points en vert pour Harris
    img_points[np.logical_and(~mask_XOR, mask_Harris_Gauss), 1] = 255 
    img_points[np.logical_and(~mask_XOR, mask_Harris_Gauss), 2] = 0

    return img_points