import cv2
import numpy as np

"""using ORB"""

def align_images(master_img, daily_img, max_feature = 5000, match_ratio=0.75, ransac_thres=0.5):
    master_grey = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    daily_grey = cv2.cvtColor(daily_img, cv2.COLOR_BGR2GRAY)

    #initialize ORB
    orb = cv2.ORB_create(max_feature)
    keypoints_daily, descriptors_daily = orb.detectAndCompute(daily_grey, None)
    keypoints_master, descriptors_master = orb.detectAndCompute(master_grey,None)

    if descriptors_daily is None or descriptors_master is None:
        raise ValueError("Failed to compute descriptors. Image may be blank or severely blurred.")
    
    # match the feature brute force with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors_daily, descriptors_master, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 20:
        raise ValueError(f"Alignment failed: Only {len(good_matches)} valid features found. Prompt worker to retake photo.")

    points_daily = np.float32([keypoints_daily[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_master = np.float32([keypoints_master[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(points_daily, points_master, cv2.RANSAC, ransac_thres)

    if matrix is None:
        raise ValueError("Homography calculation failed. The perspective difference is too extreme.")

    height, width = master_img.shape[:2]
    aligned_img = cv2.warpPerspective(daily_img, matrix, (width, height))

    return aligned_img

    




   