import numpy as np
from skimage import io,color,measure

def get_noise_3x3(img: object, i: object, j: object) -> object:
    # Wc0 va stoca valorile pixelilor care nu sunt zgomot de intensitate zero
    Wc0 = []
    # Wc1 va stoca valorile pixelilor care nu sunt zgomot de intensitate maxima
    Wc1 = []

    # Extrage fereastra de 3x3 pixeli in jurul pixelului (i, j) si o transforma intr-un vector unidimensional
    Wc = img[i - 1:i + 2, j - 1:j + 2].ravel()

    # Parcurge fiecare valoare din fereastra de 3x3
    for k in Wc:
        # Daca valoarea pixelului nu este 0, atunci adaugam valoarea in lista Wc0
        if k != 0:
            Wc0.append(k)

    # Parcurge valorile din Wc0
    for y in Wc0:
        # Dacă valoarea pixelului nu este 1, adauga valoarea in lista Wc1
        if y != 1:
            Wc1.append(y)

    # Returneaza lista Wc1 care contine valorile pixelilor care nu sunt considerate ca fiind corupte de zgomot
    return Wc1

def get_RGB(img, cR, cG, cB):
    # Functia creeaza o imagine color din trei canale de culoari separate (Roșu, Verde, Albastru)
    M = np.shape(img)[0]  # Obtine numarul de randuri din imagine
    N = np.shape(img)[1]  # Obtine numarul de coloane din imagine

    # Initializeaza o matrice de zerouri cu dimensiunea imaginii si 3 canale (pentru Roșu, Verde, Albastru)
    imagine_filtrata = np.zeros([M, N, 3])

    # Seteaza fiecare canal de culoare corespunzator in imaginea filtrata
    imagine_filtrata[:, :, 0] = cR  # Canalul Roșu
    imagine_filtrata[:, :, 1] = cG  # Canalul Verde
    imagine_filtrata[:, :, 2] = cB  # Canalul Albastru

    return imagine_filtrata  # Returneaza imaginea color compusă

