

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from skimage.io import imread,imshow
from skimage.color import rgb2gray
import cv2


""" Fonction de détection de Harris avec une fenêtre rectangulaire """

def detection_harris_rectangle(img_path,rot,window_size=3,k=0.02):   
    img = imread(img_path)
    if rot != 0 :
        img= ndi.rotate(img,rot, reshape=False)
        
    img_gray=rgb2gray(img) # ---> rgb2gray normalise la matrice

    Ix, Iy = np.gradient(img_gray)  # gradients dans les directions x et y 

    Ix2= np.square(Ix) # les trois calculs pour la matrice M
    Iy2=np.square(Iy)
    IxIy= (Ix*Iy)

    img_copy_for_corners = np.copy(img) # on crée une copie pour ne pas altérer l'imgae d'originie
    count = 0

    height, width = img_gray.shape

    
    Ix2_sum = ndi.convolve(Ix2, np.ones((window_size, window_size))) # on calcule la somme des carrés des dérivées à chaque pixel
    Iy2_sum = ndi.convolve(Iy2, np.ones((window_size, window_size)))
    IxIy_sum = ndi.convolve(IxIy, np.ones((window_size, window_size)))

    det_M = Ix2_sum * Iy2_sum - IxIy_sum ** 2  #on calcule le determinant et la trace de la matrice M à chaque pixel
    trace_M = Ix2_sum + Iy2_sum

    C = det_M - k * trace_M ** 2

    #threshold = 0.01 * harris_response.max() # on seuil la réponse pour avoir les points d'intérêts
    threshold = 0.01 * C.max() # on seuil la réponse pour avoir les points d'intérêts
    for x in range(height):
        for y in range(width):
            if C[x, y] > threshold:
                img_copy_for_corners[x, y] = [255, 0, 0] # on met en rouge les points d'intérêts
                count += 1

    return img_copy_for_corners, count



""" Fonction de suppression des non-maximas locaux """

def sup_non_maxima_optimized(img: np.ndarray, img_originale_path: str) -> np.ndarray:
    
    # On lit l'image originale
    img_originale = imread(img_originale_path)
    # On convertit l'image en niveaux de gris
    img_gray = rgb2gray(img)
    # On crée une copie de l'image avec les points d'intérêts
    image_avec_pt = np.copy(img)

    # On récupère la largeur et la hauteur de l'image
    width = np.shape(img_gray)[1]
    height = np.shape(img_gray)[0]

    # On crée un kernel de 3x3 avec des 1 partout sauf au centre
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    # On calcule le maximum des pixels voisins pour chaque pixel de l'image
    max_neighborhood = np.max(np.stack([ndi.maximum_filter(img_gray, footprint=kernel, mode='constant', cval=0.0),
                                        ndi.maximum_filter(img_gray, footprint=kernel.T, mode='constant', cval=0.0)]), axis=0)

    # On remplace les pixels qui ne sont pas des maximums locaux par les pixels de l'image originale
    image_avec_pt[max_neighborhood > img_gray] = img_originale[max_neighborhood > img_gray]

    # On compte le nombre de points d'intérêts supprimés
    count_suppr = np.sum(image_avec_pt[:, :, 0] == 255)

    return image_avec_pt, count_suppr



""" Fonction de détection de Harris avec une fenêtre gaussienne """

def detection_harris_gauss_optimized(img_path, rot,window_size=3,k=0.04,sigma=2):
    img = imread(img_path)
    if rot != 0:
        img = ndi.rotate(img, rot, reshape=False)
    img_gray = rgb2gray(img)

    width = np.shape(img_gray)[1]  # largeur et hauteurs de l'image
    height = np.shape(img_gray)[0]
   
    Ix, Iy = np.gradient(img_gray)  # gradients dans les directions x et y 

    Ix2 = np.square(Ix)  # les trois calculs pour la matrice M
    Iy2 = np.square(Iy)
    IxIy = (Ix * Iy)

    
    kernel_size = window_size # Adjust the kernel size as needed
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    for y in range(kernel_size):
        for x in range(kernel_size):
            gaussian_kernel[y, x] = 1 / (2 * np.pi * sigma ** 2) * np.exp(
                -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2))

    # Convolve the matrices with the Gaussian kernel
    Ix2gauss = sig.convolve2d(Ix2, gaussian_kernel, boundary='fill', mode='same')
    Iy2gauss = sig.convolve2d(Iy2, gaussian_kernel, boundary='fill', mode='same')
    IxIygauss = sig.convolve2d(IxIy, gaussian_kernel, boundary='fill', mode='same')

    img_copy_for_corners = np.copy(img)  # on créer une copie pour ne pas altérer l'image d'origine
    count = 0

    deter = (Ix2gauss * Iy2gauss) - (IxIygauss ** 2)
    trace = Ix2gauss + Iy2gauss
    C = deter - k * (trace ** 2)  # calcul du critère pour le seuillage des points d'intérêts
    threshold = 0.01 * C.max()
    img_copy_for_corners[C > threshold] = [255, 0, 0]  # on met en rouge les points d'intérêts
    count = np.sum(C > threshold)

    return img_copy_for_corners, count



""" Fonction de détection de Harris avec une Hessienne """

def detection_hessienne(img_path, rot):
    img = imread(img_path)
    if rot != 0:
        img = ndi.rotate(img, rot, reshape=False)

    img_gray = rgb2gray(img)

    Ix, Iy = np.gradient(img_gray)  # gradient in x and y directions
    Ixx,Ixy = np.gradient(Ix)
    Iyx,Iyy = np.gradient(Iy)

    img_copy_for_corners = np.copy(img)
    count = 0

    height, width = img_gray.shape

    det_H = Ixx * Iyy - Ixy* Iyx

    C = det_H

    threshold = 0.01 * C.max()  # seuil

    for x in range(height):
        for y in range(width):
            if C[x, y] > threshold:
                img_copy_for_corners[x, y] = [255, 0, 0]
                count += 1

    return img_copy_for_corners, count


