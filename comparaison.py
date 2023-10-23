
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
import time
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

from fast import fast_1
from harris import detection_harris_gauss_optimized
from harris import sup_non_maxima_optimized




""" Fonction pour comparer les points d'intérêt de FAST et Harris Gauss dans leur répartition sur l'image """

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


""" Comparaison des points d'intérêt de FAST et Harris Gauss au niveau de la précision de sélection """

def comp_quotient(img_path):

    img = imread(img_path)

    # On définit les paramètres pour FAST et Harris Gauss
    radius_FAST = 12
    threshold_Harris_Gauss = 0.04
    sigma_Harris_Gauss = 2

    # on définit les valeurs de k pour FAST et n pour Harris Gauss
    k_fast_values = np.arange(0.01, 0.11, 0.01)
    n_harris_values = np.arange(3, 20, 1)

    # Initialisation des listes pour stocker les quotients pour chaque méthode
    quotient_FAST = []
    quotient_Harris_Gauss = []

    # Boucle sur les valeurs de k et n pour calculer les quotients pour chaque méthode
    for k_f in k_fast_values:
        img_FAST, count_fast = fast_1(img_path, k_f, radius_FAST, 0)
        # Suppression des non-maxima locaux
        _, count_suppr_fast = sup_non_maxima_optimized(img_FAST, img_path)
        # Compte le nombre de points d'intérêt pour chaque méthode
        num_FAST = count_fast
        # Compte le nombre de points d'intérêt post suppression des non-maxima locaux pour chaque méthode
        num_FAST_post_sup = count_fast - count_suppr_fast
        # Calcule le quotient pour chaque méthode
        quotient_FAST.append(num_FAST_post_sup / num_FAST)
    

    for n_h in n_harris_values :
        img_Harris_Gauss, count_harris = detection_harris_gauss_optimized(img_path, 0, n_h, 0.04, 2)
        # Suppression des non-maxima locaux
        _, count_suppr_harris = sup_non_maxima_optimized(img_Harris_Gauss, img_path)
        # Compte le nombre de points d'intérêt pour chaque méthode
        num_Harris_Gauss = count_harris
        # Compte le nombre de points d'intérêt post suppression des non-maxima locaux pour chaque méthode
        num_Harris_Gauss_post_sup = count_harris - count_suppr_harris
        # Calcule le quotient pour chaque méthode
        quotient_Harris_Gauss.append(num_Harris_Gauss_post_sup / num_Harris_Gauss)

    return quotient_FAST, quotient_Harris_Gauss, k_fast_values, n_harris_values


def comp_openCV(img_path):
   
    img = plt.imread(img_path)

    # Définition des paramètres pour FAST et Harris Gauss
    threshold_FAST = 0.05
    radius_FAST = 12
    threshold_Harris_Gauss = 0.04
    sigma_Harris_Gauss = 2

    # Définition des valeurs de k pour FAST et n pour Harris Gauss
    k_values = np.arange(0, 1, 0.1)

    # Initialisation des listes pour stocker les temps d'exécution pour chaque méthode
    time_cv2_Harris = []
    time_cv2_FAST = []
    time_Harris_Gauss = []
    time_FAST = []

    # Boucle sur les valeurs de k et n pour calculer les temps d'exécution pour chaque méthode
    for k in k_values:
        # détection des points d'intérêt avec OpenCV
        start_time = time.time()
        dst_cv2_Harris = cv2.cornerHarris(np.uint8(rgb2gray(img)*255), 2, 3, k, cv2.BORDER_DEFAULT)
        time_cv2_Harris.append(time.time() - start_time)

        start_time = time.time()
        #dst_cv2_FAST = cv2.FastFeatureDetector_create(int(threshold_FAST*100))(np.uint8(rgb2gray(img)*255), None)
        fast = cv2.FastFeatureDetector_create(int(threshold_FAST*100))
        kp_FAST = fast.detect(np.uint8(rgb2gray(img)*255), None)
        dst_cv2_FAST = np.zeros_like(img)
        dst_cv2_FAST = cv2.drawKeypoints(img, kp_FAST, dst_cv2_FAST, color=(255,0,0))
        time_cv2_FAST.append(time.time() - start_time)

        # détection des points d'intérêt avec notre méthode
        start_time = time.time()
        img_Harris_Gauss, _ = detection_harris_gauss_optimized("M1.JPG", 0, 3, k, sigma_Harris_Gauss)
        time_Harris_Gauss.append(time.time() - start_time)

        start_time = time.time()
        img_FAST, _ = fast_1("M1.JPG", k, radius_FAST, 0)
        time_FAST.append(time.time() - start_time)

    return time_cv2_Harris, time_cv2_FAST, time_Harris_Gauss, time_FAST, k_values