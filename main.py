import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.align_images import *

master_path  = os.path.join("images", "master.jpg")
daily_path = os.path.join("images","daily.jpg")

master_image = cv2.imread(master_path)
daily_image = cv2.imread(daily_path)


try:
    # Run the alignment
    aligned_daily_image =  align_images(master_image, daily_image)

    # Plot the results side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(master_image)
    axes[0].set_title("Master Image (Target)")
    axes[0].axis('off')
    
    axes[1].imshow(daily_image)
    axes[1].set_title("Daily Image (Original Unaligned)")
    axes[1].axis('off')
    
    axes[2].imshow(aligned_daily_image)
    axes[2].set_title("Daily Image (Warped & Aligned)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"Alignment Failed: {e}")