import matplotlib.pyplot as plt
from skimage import io
from skimage.util import random_noise
import numpy as np
from Filters import get_images, filters
from MAE_SSIM import meanErr, structuralSim
from Utils import get_RGB
from MSE_PSNR import psnr, mse

# Se încarca o imagine folosind biblioteca skimage
img = io.imread('Penguins.jpg')

# Se adauga zgomot impulsiv (sare și piper) ssi gaussian imaginii
noise = 0.1  # Setează nivelul de zgomot
img_impulse_noise = random_noise(img, mode='s&p', amount=noise)  # Utilizare directa a lui random_noise
img_gaussian_noise = random_noise(img, mode='gaussian', var=noise)  # Utilizare directa a lui random_noise


# Separam canalele de culoare ale imaginilor zgomotoase in componente Rosu, Verde și Albastru
R_imp = img_impulse_noise[:, :, 0]  # Canalul Rosu pentru zgomotul impulsiv
G_imp = img_impulse_noise[:, :, 1]  # Canalul Verde pentru zgomotul impulsiv
B_imp = img_impulse_noise[:, :, 2]  # Canalul Albastru pentru zgomotul impulsiv

R_gau = img_gaussian_noise[:, :, 0]  # Canalul Rosu pentru zgomotul gaussian
G_gau = img_gaussian_noise[:, :, 1]  # Canalul Verde pentru zgomotul gaussian
B_gau = img_gaussian_noise[:, :, 2]  # Canalul Albastru pentru zgomotul gaussian

# Aplicam diverse filtre pe fiecare canal de culoare al ambelor tipuri de zgomot
images = get_images(R_imp, R_gau)
gaussian_noise_image_filtered_R = images[0]
impulse_noise_image_filtered_R = images[1]
gausian_noise_median_filter_R = images[2]
impulse_noise_median_filter_R = images[3]
gausian_noise_mean_filter_R = images[4]
impulse_noise_mean_filter_R = images[5]

# Repeta procesul pentru canalul Verde
images = get_images(G_imp, G_gau)
gaussian_noise_image_filtered_G = images[0]
impulse_noise_image_filtered_G = images[1]
gausian_noise_median_filter_G = images[2]
impulse_noise_median_filter_G = images[3]
gausian_noise_mean_filter_G = images[4]
impulse_noise_mean_filter_G = images[5]

# Repeta procesul pentru canalul Albastru
images = get_images(B_imp, B_gau)
gaussian_noise_image_filtered_B = images[0]
impulse_noise_image_filtered_B = images[1]
gausian_noise_median_filter_B = images[2]
impulse_noise_median_filter_B = images[3]
gausian_noise_mean_filter_B = images[4]
impulse_noise_mean_filter_B = images[5]

'''Afisare imagini'''
# Crearea imaginilor filtrate din canalele de culoare separate

# Generarea imaginii cu zgomot gaussian filtrat combinand canalele filtrate R, G și B
img_zgomot_gaussian_filtrata = get_RGB(img_gaussian_noise, gaussian_noise_image_filtered_R,
                                       gaussian_noise_image_filtered_G, gaussian_noise_image_filtered_B)

# Generarea imaginii cu zgomot impulsiv filtrat combinand canalele filtrate R, G și B
img_zgomot_impulsiv_filtrata = get_RGB(img_impulse_noise, impulse_noise_image_filtered_R,
                                       impulse_noise_image_filtered_G, impulse_noise_image_filtered_B)

# Generarea imaginii cu zgomot gaussian și filtru median aplicat, combinand canalele R, G și B
img_zgomot_gaussian_filtrata_median = get_RGB(img_gaussian_noise, gausian_noise_median_filter_R,
                                              gausian_noise_median_filter_G, gausian_noise_median_filter_B)

# Generarea imaginii cu zgomot impulsiv și filtru median aplicat, combinand canalele R, G și B
img_zgomot_impulsiv_filtrata_median = get_RGB(img_impulse_noise, impulse_noise_median_filter_R,
                                              impulse_noise_median_filter_G, impulse_noise_median_filter_B)

# Generarea imaginii cu zgomot gaussian și filtru de mediere aplicat, combinand canalele R, G și B
img_zgomot_gaussian_filtrata_mean = get_RGB(img_gaussian_noise, gausian_noise_mean_filter_R,
                                            gausian_noise_mean_filter_G, gausian_noise_mean_filter_B)

# Generarea imaginii cu zgomot impulsiv și filtru de mediere aplicat, combinand canalele R, G și B
img_zgomot_impulsiv_filtrata_mean = get_RGB(img_impulse_noise, impulse_noise_mean_filter_R, impulse_noise_mean_filter_G,
                                            impulse_noise_mean_filter_B)

# Crearea primei figuri pentru imaginea originala
fig1 = plt.figure(figsize=(10, 5))
plt.imshow(img)  # Afișează imaginea originală
plt.axis('off')  # Dezactivează axele
plt.title("Imagine Originala", fontsize=10)  # Adaugă un titlu subgraficului

# Crem a 2-a figura pentru imaginile cu zgomot Gaussian

fig2 = plt.figure(figsize=(10, 5))
# Adaugă un subgrafic pentru imaginea cu zgomot gaussian
fig2.add_subplot(2, 2, 1)  # Poziționarea subgraficului
plt.imshow(img_gaussian_noise)  # Afișează imaginea cu zgomot gaussian
plt.axis('off')  # Dezactivează axele
x = float(structuralSim(img,img_gaussian_noise))
x = round(x,2)
plt.title("Imagine Zgomot Gaussian cu similaritate: " + str(x) , fontsize=10)  # Adaugă un titlu subgraficului


# Adaugă un subgrafic pentru imaginea gaussiană filtrată
fig2.add_subplot(2, 2, 2)  # Poziționarea subgraficului
plt.imshow(img_zgomot_gaussian_filtrata)  # Afișează imaginea gaussiană filtrată
plt.axis('off')  # Dezactivează axele
x = float(structuralSim(img, img_zgomot_gaussian_filtrata))  # Calculează SSIM pentru imaginea filtrată
x = round(x, 2)  # Rotunjește valoarea SSIM
plt.title("Imagine zgomot Gaussian filtrata cu similaritate: " + str(x), fontsize=10)  # Adaugă un titlu cu SSIM

# Adaugă un subgrafic pentru imaginea gaussiană cu filtru median aplicat
fig2.add_subplot(2, 2, 3)
plt.imshow(img_zgomot_gaussian_filtrata_median)
plt.axis('off')
x = float(structuralSim(img, img_zgomot_gaussian_filtrata_median))
x = round(x, 2)
plt.title("Imagine zgomot Gaussian - median cu similaritate: " + str(x), fontsize=10)

# Adaugă un subgrafic pentru imaginea cu zgomot gaussian și filtru de mediere aplicat
fig2.add_subplot(2, 2, 4)
plt.imshow(img_zgomot_gaussian_filtrata_mean)
plt.axis('off')
x = float(structuralSim(img, img_zgomot_gaussian_filtrata_mean))
x = round(x, 2)
plt.title("Imagine zgomot Gaussian - mediere cu similaritate: " + str(x), fontsize=10)

# Crem a 3-a figura pentru imaginile cu zgomot impulsiv
fig3 = plt.figure(figsize=(10, 5))


# Adaugă un subgrafic pentru imaginea cu zgomot impulsiv
fig3.add_subplot(2, 2, 1)
plt.imshow(img_impulse_noise)
plt.axis('off')
x = float(structuralSim(img, img_impulse_noise))
x = round(x, 2)
plt.title("Imagine zgomot impulsiv cu similaritate: " + str(x), fontsize=10)

# Adaugă un subgrafic pentru imaginea cu zgomot impulsiv filtrată
fig3.add_subplot(2, 2, 2)
plt.imshow(img_zgomot_impulsiv_filtrata)
plt.axis('off')
x = float(structuralSim(img, img_zgomot_impulsiv_filtrata))
x = round(x, 2)
plt.title("Imagine zgomot impulsiv filtrata cu similaritate: " + str(x), fontsize=10)


# Adaugă un subgrafic pentru imaginea cu zgomot impulsiv și filtru median aplicat
fig3.add_subplot(2, 2, 3)
plt.imshow(img_zgomot_impulsiv_filtrata_median)
plt.axis('off')
x = float(structuralSim(img, img_zgomot_impulsiv_filtrata_median))
x = round(x, 2)
plt.title("Imagine zgomot impulsiv -  median cu similaritate: " + str(x), fontsize=10)

# Adaugă un subgrafic pentru imaginea cu zgomot gaussian și filtru de mediere aplicat
fig3.add_subplot(2, 2, 4)
plt.imshow(img_zgomot_impulsiv_filtrata_mean)
plt.axis('off')
x = float(structuralSim(img,img_zgomot_impulsiv_filtrata_mean))
x = round(x,2)
plt.title("Imagine zgomot impulsiv - mediere cu similaritate: " + str(x),fontsize =10)


plt.show()

img_float = img.astype('float64')
img_impulse_noise_float = img_impulse_noise.astype('float64')
img_impulse_noise_filtrata_float = img_impulse_noise.astype('float64')
img_impulse_noise_filtrata_median_float = img_impulse_noise.astype('float64')
img_impulse_noise_filtrata_mean_float = img_impulse_noise.astype('float64')

# Pentru MSE SI PSNR

# Pentru MAE SI SSIM
print("Imagine zgomot impulsiv:")
print("MAE: " + str(meanErr(img, img_impulse_noise)))
print("SSIM: " + str(structuralSim(img, img_impulse_noise)))
print("MSE: " + str(mse(img_float, img_impulse_noise_float)))
print("PNSR: " + str(psnr(img_float, img_impulse_noise_float)))
print()

print("Imagine zgomot impulsiv filtrata :")
print("MAE: " + str(meanErr(img, img_zgomot_impulsiv_filtrata)))
print("SSIM: " + str(structuralSim(img, img_zgomot_impulsiv_filtrata)))
print("MSE: " + str(mse(img_float, img_impulse_noise_filtrata_float)))
print("PNSR: " + str(psnr(img_float, img_impulse_noise_filtrata_float)))
print()

print("Imagine zgomot impulsiv filtrata - median:")
print("MAE: " + str(meanErr(img, img_zgomot_impulsiv_filtrata_median)))
print("SSIM: " + str(structuralSim(img, img_zgomot_impulsiv_filtrata_median)))
print("MSE: " + str(mse(img_float, img_impulse_noise_filtrata_median_float)))
print("PNSR: " + str(psnr(img_float, img_impulse_noise_filtrata_median_float)))
print()

print("Imagine zgomot impulsiv filtrata - mediere:")
print("MAE: " + str(meanErr(img, img_zgomot_impulsiv_filtrata_mean)))
print("SSIM: " + str(structuralSim(img, img_zgomot_impulsiv_filtrata_mean)))
print("MSE: " + str(mse(img_float, img_impulse_noise_filtrata_mean_float)))
print("PNSR: " + str(psnr(img_float, img_impulse_noise_filtrata_mean_float)))