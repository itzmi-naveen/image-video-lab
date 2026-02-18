from google.colab import files
files.upload()

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def rotate_image(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))


def scale_image(image, scale_x, scale_y):
    return cv2.resize(image, None, fx=scale_x, fy=scale_y)


def skew_image(image, skew_x, skew_y):
    h, w = image.shape[:2]
    matrix = np.float32([
        [1, skew_x, 0],
        [skew_y, 1, 0]
    ])
    return cv2.warpAffine(image, matrix, (w, h))


def affine_transform(image, pts_src, pts_dst):
    matrix = cv2.getAffineTransform(pts_src, pts_dst)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def bilinear_transform(image, pts_src, pts_dst):
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))


image = cv2.imread("img.jpg")

if image is None:
    raise ValueError("Image not loaded. Please upload img.jpg")



rotated = rotate_image(image, 45)
cv2_imshow(rotated)

scaled = scale_image(image, 1.5, 1.5)
cv2_imshow(scaled)

skewed = skew_image(image, 0.2, 0.1)
cv2_imshow(skewed)

src_affine = np.float32([[50, 50], [200, 50], [50, 200]])
dst_affine = np.float32([[10, 100], [200, 50], [100, 250]])
affine_img = affine_transform(image, src_affine, dst_affine)
cv2_imshow(affine_img)

src_persp = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
dst_persp = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
perspective_img = bilinear_transform(image, src_persp, dst_persp)
cv2_imshow(perspective_img)

# Output :
<img width="386" height="457" alt="image" src="https://github.com/user-attachments/assets/0a4615f9-4983-4860-b61d-6d7736c64502" />
<img width="235" height="199" alt="image" src="https://github.com/user-attachments/assets/cdf4362d-afdb-4f91-af01-54bd1995d88b" />
