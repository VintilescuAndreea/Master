import numpy as np
import math
from Utils import get_noise_3x3
from scipy import signal, ndimage

def filters(nImg):
    # Modul de parcurgere al diagramei
    Nd = noiseDensity_Nd(nImg)
    if Nd < 0.45:
        nImg = IEHCLND(nImg, Nd)
    I1, I2 = nImg, nImg
    Layers = ['Max', 'Min', 'Min', 'Max']
    for Layer in Layers:
        I1, I2 = CMMP(I1, I2, Layer)
    OutImg = RandS(I1, I2, nImg)
    return OutImg

def noiseDensity_Nd(f):
    # Determina densitatea zgomotului intr-o imagine
    M = f.shape[0]  # Numarul de randuri din imagine
    N = f.shape[1]  # Numarul de coloane din imagine
    noise_mask = np.ones([M, N])  # Initializeaza o masca de zgomot cu valori de 1

    # Parcurge fiecare pixel din imagine
    for i in range(M):
        for j in range(N):
            # Setareaza valoarea in noise_mask la 0 daca pixelul este zgomotos,
            # adica daca intensitatea este 0 (negru) sau 255 (alb), ca în Ec. (1).
            if f[i][j] == 0 or f[i][j] == 255:
                noise_mask[i][j] = 0

    suma = 0  # Initializeaza suma pentru a calcula numarul total de pixeli fara zgomot
    # Calculeaza suma pixelilor fara zgomot
    for i in range(M):
        for j in range(N):
            suma += noise_mask[i][j]

    Nd = suma / (M * N)  # Calculeaza densitatea zgomotului ca raport dintre pixelii nezgomotosi si totalul de pixeli
    return Nd  # Returneaza densitatea zgomotului

def IEHCLND(img, Nd):
    # Functia aplica un filtru pentru reducerea zgomotului impulsiv din imaginea 'img' si se activeaza numai la Nd< 0.45
    img = np.pad(img, pad_width=1, mode='symmetric')  # Extindem imaginea cu o margine simetrica pentru a procesa pixelii

    alpha = math.floor(Nd / 0.1)  # Calculeaza un prag de informatie 'alpha' bazat pe densitatea zgomotului 'Nd'
    M = img.shape[0]  # Numarul de randuri din imaginea extinsa
    N = img.shape[1]  # Numarul de coloane din imaginea extinsa

    oImg = img  # Initializeaza imaginea de ieșire ca o copie a imaginii de intrare

    # Parcurge fiecare pixel din imagine, excluzand marginile adaugate prin padding
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Verifica daca pixelul curent este considerat zgomot
            if img[i][j] == 0 or img[i][j] == 1:
                Wc = get_noise_3x3(img, i, j)  # Obtine o fereastra fara zgomot in jurul pixelului curent
                if len(Wc) > alpha:  # Verifica daca numarul de pixeli nezgomotosi este mai mare decat pragul impus
                    oImg[i][j] = np.median(Wc)  # Inlocuim pixelul corupt cu valoarea mediana a ferestrei nezgomotoase

    oImg = oImg[1:M - 1, 1:N - 1]  # Elimina padding-ul adaugat la inceput

    return oImg  # Returneaza imaginea procesata


def CMMP(I1, I2, str):
    # Extindem imaginile cu o margine simetrica pentru a gestiona pixelii de la marginea imaginii
    I1 = np.pad(I1, pad_width=1, mode='symmetric')
    I2 = np.pad(I2, pad_width=1, mode='symmetric')

    # Obtinem dimensiunile imaginilor extinse
    M, N = I1.shape

    # Initializam imaginile de iesire ca o copie a imaginilor de intrare
    O1, O2 = I1.copy(), I2.copy()

    # Parcurgem fiecare pixel din imaginea extinsa, excluzand marginea adaugata
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Verificam dacă pixelul curent are valoarea 0 sau 1
            if I1[i][j] == 0 or I1[i][j] == 1:
                # Obtinem fereastra de 3x3 pixeli in jurul pixelului curent si eliminam valorile 0 si 1
                Wc = get_noise_3x3(I1, i, j)
                try:
                    Wc.remove(1)  # incercam să eliminam valoarea 1 din fereastra, dacă exista
                    Wc.remove(0)  # incercam să eliminam valoarea 0 din fereastra, dacă exista
                except ValueError:
                    # Daca valorile nu sunt gasite in lista, ignoram eroarea
                    pass

                # Daca fereastra nu este goală dupa eliminarea zgomotului
                if len(Wc) > 0:
                    # in funcție de modul specificat prin 'str', aplicam pooling-ul min sau max
                    if str == 'Max':
                        O1[i][j] = np.max(Wc)  # Max pooling pentru I1
                        O2[i][j] = np.min(Wc)  # Min pooling pentru I2
                    elif str == 'Min':
                        O1[i][j] = np.min(Wc)  # Min pooling pentru I1
                        O2[i][j] = np.max(Wc)  # Max pooling pentru I2

    # Eliminam marginea adaugata si returnam imaginile procesate
    O1 = O1[1:M - 1, 1:N - 1]
    O2 = O2[1:M - 1, 1:N - 1]
    return O1, O2


def RandS(I1, I2, nImg):
    # Extindem imaginile cu o margine simetrica pentru a gestiona pixelii de la marginea imaginii
    I1 = np.pad(I1, pad_width=1, mode='symmetric')
    I2 = np.pad(I2, pad_width=1, mode='symmetric')
    nImg = np.pad(nImg, pad_width=1, mode='symmetric')

    # Obtinem dimensiunile imaginii zgomotoase extinse
    M, N = nImg.shape

    # Calciulam media intensitatii pixel cu pixel a celor două imagini de intrare - recombinare
    oImg = (I1 + I2) / 2

    # Parcurgem fiecare pixel din imaginea zgomotoasa extinsa, excluzand marginea adaugata
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Verificam dacă pixelul curent este corupt
            if nImg[i][j] == 0 or nImg[i][j] == 1:
                # Obtinem fereastra de 3x3 pixeli în jurul pixelului curent din imaginea rezultata din medie
                Wc = get_noise_3x3(oImg, i, j)
                # Inlocuim pixelul corupt cu media valorilor din fereastra fară zgomot
                oImg[i][j] = np.mean(Wc)

    # Eliminam marginea adaugata și returnam imaginea procesata
    oImg = oImg[1:M - 1, 1:N - 1]
    return oImg


def mean_filter(img):
    med = signal.convolve2d(img, np.ones([3, 3]) / 9, mode='same', boundary='symm')
    return med

def get_images(imp_noise, gaus_noise):
    gaussian_noise_image_filtered = filters(gaus_noise)
    impulse_noise_image_filtered = filters(imp_noise)
    gausian_noise_median_filter = ndimage.median_filter(gaus_noise, (3, 3))
    impulse_noise_median_filter = ndimage.median_filter(imp_noise, (3, 3))
    gausian_noise_mean_filter = mean_filter(gaus_noise)
    impulse_noise_mean_filter = mean_filter(imp_noise)
    return (gaussian_noise_image_filtered, impulse_noise_image_filtered,
            gausian_noise_median_filter, impulse_noise_median_filter,
            gausian_noise_mean_filter, impulse_noise_mean_filter)
