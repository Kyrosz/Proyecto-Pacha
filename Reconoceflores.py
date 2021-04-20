#Leer imagen / rgb y grises / .shape()

from matplotlib import pyplot as plt
import numpy as np
import cv2

img_path = 'ave5.jpg'
img = cv2.imread(r'C:\Users\jandr\Desktop\Robotica\imagenes\\' + img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_grises = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

print('type(img):', type(img))
print('img.shape:', img.shape)
print('img.dtype:', img.dtype)

plt.figure()
plt.imshow(img)
plt.axis('off')

#%% Canales HSV
for i in range(img_hsv.shape[2]):
    plt.figure()
    plt.imshow(img_hsv[:, :, i], cmap='jet')
    plt.axis('off')

#%% Histograma

plt.figure(figsize=(15,5))
plt.title('Histograma')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

_ = plt.hist(img, 510)

#%% Mascara y filtro inrange

Mascara = cv2.inRange(img_grises, 0, 120)/255

plt.figure()
plt.imshow(Mascara, cmap='gray')
plt.axis('off')

img_mascara = np.zeros_like(img)

for i in range(img.shape[2]):
  img_mascara[:,:,i] = np.multiply(img[:,:,i], Mascara)

plt.figure()
plt.imshow(img_mascara)
plt.axis('off')

#%% Canales

for i in range(img.shape[2]):
    plt.figure(figsize=(15,5))
    plt.title('Histograma RGB (Canal %i)' % i)
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')

    img_histograma = plt.hist(img[:, :, i].flatten(), 510)

    plt.figure()
    plt.imshow(img[:, :, i],cmap='jet')
    plt.axis('off')
    
#%% Operador lineal (scaling)

img_lineal = np.float32(img)

v_min = np.min(img_lineal)
print('valor minimo -> ', v_min)
v_max = np.max(img_lineal)
print('valor maximo -> ', v_max)

for i in range(img_lineal.shape[0]):
  for j in range(img_lineal.shape[1]):
      img_lineal[i, j] = 255*(img_lineal[i, j] - v_min)/(v_max - v_min)

img_lineal = np.uint8(img_lineal)

plt.figure()
plt.imshow(img_lineal, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

#%% Operador logaritmo

img_log = np.float32(img)

v_min = np.min(img_log)
print('valor minimo -> ', v_min)
v_max = np.max(img_log)
print('valor maximo -> ', v_max)

for i in range(img_log.shape[0]):
  for j in range(img_log.shape[1]):
      img_log[i, j] = (img_log[i, j] - v_min)/(v_max - v_min)

alpha = 0.5

for i in range(img_log.shape[0]):
  for j in range(img_log.shape[1]):
    img_log[i, j] = np.log(img_log[i, j] + alpha)

v_min = np.min(img_log)
print('valor minimo log -> ', v_min)
v_max = np.max(img_log)
print('valor maximo log -> ', v_max)

for i in range(img_log.shape[0]):
  for j in range(img_log.shape[1]):
    img_log[i, j] = 255*(img_log[i, j] - v_min)/(v_max - v_min)

img_log = np.uint8(img_log)

fig = plt.figure()
plt.imshow(img_log, cmap='gray')
plt.axis('off')

#%% Tratamiento HSV / filtro gaussiano

low_hsv = np.array([20, 20, 20])
up_hsv = np.array([180, 180, 180])

mascara = cv2.inRange(img_hsv, low_hsv, up_hsv)

plt.figure()
plt.imshow(mascara, cmap='gray')

img_blur = cv2.GaussianBlur(img, (15, 15), 8)

plt.figure()
plt.imshow(img_blur)

hsv_blur = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

mascara = cv2.inRange(hsv_blur, low_hsv, up_hsv)

plt.figure()

plt.imshow(mascara, cmap='gray')



