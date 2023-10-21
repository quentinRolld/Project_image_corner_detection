import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
from skimage.io import imread,imshow
from skimage.color import rgb2gray
import cv2

def fast_for_matching(image_path: str, t: float, n: int, rot: int) -> np.ndarray:
    image = imread(image_path)

    if rot != 0 :
        image= ndi.rotate(image,rot, reshape=False)
    img_gray=rgb2gray(image)
    corners=[]
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
                corners.append([x,y])
            
            if((y+3)%width == 0):
                break
    # Return the list of all detected corners' x and y coordinates
    return corners

from scipy.spatial.distance import cdist

def bloc_descriptor2(img_path1, img_path2, block_size, threshold):
  
    # récupérer les points d'intérêt (img 1 et 2)
    vect_interet_1 = fast_for_matching(img_path1, 0.09, 12, 0) 
    vect_interet_2 = fast_for_matching(img_path2, 0.09, 12, 0) 
    taille_vect1 = np.shape(vect_interet_1)[0]
    taille_vect2 = np.shape(vect_interet_2)[0]

    # créer n1+n2 vecteurs de bloc (avec n1, n2 = le nombre de points d'intérêt dans img 1 et 2 resp)
    # On stock ces vecteurs sous forme de matrice. Les deux matrices sont vecteurs_de_bloc_1 et 2. Une matrice par image.

    vecteurs_de_bloc_1 = []
    vecteurs_de_bloc_2 = []

    img_gray1=rgb2gray(imread(img_path1))
    img_gray2=rgb2gray(imread(img_path2))

    # parcourir tous les points d'intérêt et créer les vecteurs de bloc correspondants
    for i in range(taille_vect1):
        (x,y)=vect_interet_1[i]
        intvect1 = []
        for j in range(-block_size//2, block_size//2+1):
            for k in range(-block_size//2, block_size//2+1):
                intvect1.append(img_gray1[x+j][y+k])
        vecteurs_de_bloc_1.append(intvect1)
        
    for i in range(taille_vect2):
        (x,y)=vect_interet_2[i]
        intvect2 = []
        for j in range(-block_size//2, block_size//2+1):
            for k in range(-block_size//2, block_size//2+1):
                intvect2.append(img_gray2[x+j][y+k])
        vecteurs_de_bloc_2.append(intvect2)

    # Suppression des mauvais match : appariement croisé pour chaque pt d'intérêt
    dist_matrix = cdist(vecteurs_de_bloc_1, vecteurs_de_bloc_2, metric='correlation')
    min_dist1 = np.argmin(dist_matrix, axis=1)
    min_dist2 = np.argmin(dist_matrix, axis=0)

    # Cross-matching to ensure that d(ai, bj) = min(d(ai, bk)) and d(bj, ai) = min(d(bj, ak))
    appariement = []
    for i in range(len(min_dist1)):
        if min_dist2[min_dist1[i]] == i and dist_matrix[i, min_dist1[i]] < threshold:
            appariement.append((vect_interet_1[i], vect_interet_2[min_dist1[i]]))

        # Removing points with large vector differences
    vectors_1 = np.array([vecteurs_de_bloc_1[vect_interet_1.index(match[0])] for match in appariement])
    vectors_2 = np.array([vecteurs_de_bloc_2[vect_interet_2.index(match[1])] for match in appariement])
    vector_diff = np.linalg.norm(vectors_1 - vectors_2, axis=1)
    mean_diff = np.mean(vector_diff)
    std_diff = np.std(vector_diff)
    appariement_filtered = [appariement[i] for i in range(len(appariement)) if vector_diff[i] < mean_diff + std_diff]

    
    # Extracting matching points for image 1 and image 2
    matching_points_1 = [match[0] for match in appariement_filtered]
    matching_points_2 = [match[1] for match in appariement_filtered]

    return matching_points_1, matching_points_2



