

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from skimage.io import imread,imshow
from skimage.color import rgb2gray
import cv2


# On commence par implémenter une fonction qui récupère la valeur de l'intensité des 16 pixels du cercle autour du pixel p_0 que l'on cherche à tester.

''' Cette fonction permet de calculer les points d'un cercle de rayon 3
autour d'un point de coordonnées (ligne,col) et de renvoyer les coordonnées
de ces points dans une liste'''

def cercle(ligne: int, col: int) -> np.ndarray :
    

    pt = [(ligne-3, col),(ligne-3, col+1),(ligne-2, col+2),(ligne-1, col+3),(ligne, col+3),(ligne+1, col+3),
          (ligne+2, col+2),(ligne+3, col+1),(ligne+3, col),(ligne+3, col-1),(ligne+2, col-2),(ligne+1, col-3),
          (ligne, col-3),(ligne-1, col-3),(ligne-2, col-2),(ligne-3, col-1)]
    return pt




# Puis, on implémente la fonction capable de comparer l'intensité des différents points du cercle autour de p0.
# Si n pixels consécutifs du cercle ont une intensité suffisamment différente de celle de p0, la fonction renvoie True.

''' Cette fonction permet de comparer l'intensité du point p0 avec les points du cercle'''

def compare_Intensity(image: np.ndarray, pt_cercle: list, I_p0: float, t: float, n: int):
    sup_nb = 0

    # On commence par comparer l'intensité des pixels 1, 5, 9 et 13 avec p0 pour gagner en efficacité
    # Cela nous permet d'exclure directement une bonne partie des pixels qui ne sont pas des points d'intérêt
    for pt in (pt_cercle[0], pt_cercle[4], pt_cercle[8], pt_cercle[12]):
        if (image[pt[0]][pt[1]] > I_p0 + t) or (image[pt[0]][pt[1]] < I_p0 - t):
            sup_nb += 1     # on compte le nombre de points parmis les 4, dont l'intensité est supérieure à celle du point p0
    if sup_nb >= 3:       # si ce nombre est supérieur ou égal à 3, il s'agit peut-être d'un point d'intérêt
        # Dans ce cas, on continue la vérification
        for pt in pt_cercle:
            if (image[pt[0]][pt[1]] > t + I_p0) or (image[pt[0]][pt[1]] < I_p0 - t):
                sup_nb += 1     # on compte le nombre de points du cercle consécutifs dont l'intensité est suffisamment différente de celle du point p0
                if sup_nb >= n:
                    return True # si ce nombre est supérieur au paramètre n (à initialiser), on renvoie True, il s'agit d'un point d'intérêt
            else :
                sup_nb = 0 # on veut n points CONSECUTIFS
        return False     # si on sort de celle boucle sans avoir vu n pixels consécutifs qui vérifient l'assertion, on renvoie False, ce n'est pas un point d'intérêt
    else :
        return False  # Sinon, ce n'est pas un point d'intérêt
    


# Enfin, on implémente la fonction fast() qui utilise les deux fonctions précédentes pour chaque pixel de l'image
# (sauf ceux qui ce trouve aux extrémités de l'image et pour lesquels on ne peut pas tracer un cercle complet).
# Si compare_Intensity() renvoie True, la fonction fast() marque le point d'intérêt en rouge.

""" Méthode FAST itérative """

def fast_1(image_path: np.ndarray, t: float, n: int, rot: int):
    
    image = imread(image_path)

    if rot != 0 :
        image= ndi.rotate(image,rot, reshape=False)
    
    img_gray=rgb2gray(image)

    count = 0 # nombre de pt d'intérêt


    #x,y = neshjrid
    #xN = x +[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,1]
    #yN = y +[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3]
    #imgN = img.flatten()[xN*img.shape(1)+yN]
    


    width = np.shape(image)[1] # largeur et hauteurs de l'image
    height = np.shape(image)[0]
    x, y = 0, 0

    img_gray=rgb2gray(image)
    
    img_copy_for_FAST = np.copy(image)

    for x in range(3,height-3): 
        for y in range(3,width-3):
            
            pt_cercle = cercle(x,y)
            I_p0 = img_gray[x][y]

            if compare_Intensity(img_gray, pt_cercle, I_p0, t, n):
                img_copy_for_FAST[x][y] = [255,0,0]
                count += 1
            
            if((y+3)%width == 0):
                break
            
    
    return img_copy_for_FAST, count





""" Méthode FAST avec convolution """

def fast_2(image_path: np.ndarray, t: float, n: int, rot: int) -> np.ndarray:
    
    image = imread(image_path)

    if rot != 0 :
        image= ndi.rotate(image,rot, reshape=False)
    img_gray=rgb2gray(image)

    x,y = np.meshgrid(img_gray)

    print(len(x))
    print(len(y))
    #xN = x +[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,1]
    #yN = y +[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3]
    #imgN = img.flatten()[xN*img.shape(1)+yN]
    


    width = np.shape(image)[1] # largeur et hauteurs de l'image
    height = np.shape(image)[0]
    x, y = 0, 0

    img_gray=rgb2gray(image)
    
    img_copy_for_FAST = np.copy(image)

    for x in range(3,height-3): 
        for y in range(3,width-3):
            
            pt_cercle = cercle(x,y)
            I_p0 = img_gray[x][y]

            if compare_Intensity(img_gray, pt_cercle, I_p0, t, n):
                img_copy_for_FAST[x][y] = [255,0,0]
            
            if((y+3)%width == 0):
                break
            
    
    return img_copy_for_FAST


