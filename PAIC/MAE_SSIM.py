import numpy as np
from skimage.metrics import structural_similarity


def meanErr(img1, img2):
    # Functia calculeaza eroarea medie absoluta  intre doua imagini
    M = img1.shape[0]  # Numărul de rânduri în prima imagine
    N = img1.shape[1]  # Numărul de coloane în prima imagine

    MAE = 0  # Initializează valoarea MAE cu 0

    # Parcurge fiecare pixel din ambele imagini
    for i in range(M):
        for j in range(N):
            # Calculeaza diferența absoluta intre pixelii corespondenti din cele doua imagini
            # si adaugă această valoare la suma totala MAE
            MAE += np.float64(abs(np.float64(img1[i, j]) - np.float64(img2[i, j])))

    # Calculeaza media erorilor absolute pentru toți pixelii
    return MAE / (M * N)  # Returnează eroarea medie absolută


def structuralSim(img1, img2):
    # Functia calculeaza indicele de similaritate structurala intre două imagini

    img1 = np.uint8(img1)  # Converteste prima imagine la format uint8

    img2 = np.uint8(img2 * 255)  # Converteste a doua imagine la format uint8, scaland valorile

    # Calculeaza SSIM intre cele doua imagini
    # 'full=True' returneaza întregul rezultat, nu doar scorul
    # 'multichannel=True' indica faptul ca imaginile sunt in format color (canale multiple)
    (score, diff) = structural_similarity(img1, img2, full=True, multichannel=True, channel_axis=2)

    return str(score)  # Returneaza scorul SSIM ca sir de caractere

