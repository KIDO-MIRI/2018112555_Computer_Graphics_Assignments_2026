import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def imread_korean(path):
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

img = imread_korean('input.jpg')

if img is None:
    print("Error: Images not found")
else:
    equ = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)

    psnr_val = psnr(img, cl1)
    mse_val = mse(img, cl1)
    ssim_val = ssim(img, cl1, data_range=cl1.max() - cl1.min())

    print(f"--- 평가 결과 (Original vs CLAHE) ---")
    print(f"PSNR: {psnr_val:.2f}")
    print(f"MSE: {mse_val:.2f}")
    print(f"SSIM: {ssim_val:.2f}")

    titles = ['Original', 'Equalization', 'CLAHE']
    images = [img, equ, cl1]

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
        
        plt.subplot(2, 3, i+4)
        plt.hist(images[i].ravel(), 256, [0,256])
        plt.title(f'Hist {titles[i]}')

    plt.tight_layout()
    plt.show()