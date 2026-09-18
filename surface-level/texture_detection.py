import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_texture_baseline(masked_crop, crop_mask):
    gray = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Local Binary Patterns (LBP)
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Mask LBP to surface only
    valid_lbp_pixels = lbp[crop_mask == 255]
    
    n_bins = n_points + 2
    lbp_hist, _ = np.histogram(valid_lbp_pixels, bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6) # Normalized texture histogram
    
    # 2. GLCM Features
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
    
    glcm_features = {
        "contrast": float(graycoprops(glcm, 'contrast').mean()),
        "homogeneity": float(graycoprops(glcm, 'homogeneity').mean()),
        "energy": float(graycoprops(glcm, 'energy').mean())
    }
    
    return lbp_hist.tolist(), glcm_features