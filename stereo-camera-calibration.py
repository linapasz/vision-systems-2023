import cv2
import numpy as np
import matplotlib.pyplot as plt

# Compute disparity map using Block Matching (BM)
stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute disparity map using Semi-Global Matching (SGM)
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,
    blockSize=3,
    P1=8 * 3 * 3,
    P2=32 * 3 * 3,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# kryteria zakończenia
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# wymiary szachownicy
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# przygotowanie macierzy punktów dla obiektów: (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
objp[:, :, 2] = square_size ** 2  

# macierze do przechowywania punktów obiektów 3D i 2D z wszystkich obrazów
objpoints = []  # punkty 3d w przestrzeni (rzeczywiste)
imgpointsLeft = []  # punkty 2d w płaszczyźnie obrazu (lewy)
imgpointsRight = []  # punkty 2d w płaszczyźnie obrazu (prawy)

img_width = 640
img_height = 480
image_size = (img_width, img_height)

path = "./dataset/"
image_dir = path + "pairs/"
number_of_images = 44
for i in range(1, number_of_images):
    # wczytanie obrazu lewego
    img_left = cv2.imread(image_dir + "left_%02d.png" % i)
    if img_left is None:
        print("Failed to load image: ", image_dir + "left_%02d.png" % i)
        continue

    # wczytanie obrazu prawego
    img_right = cv2.imread(image_dir + "right_%02d.png" % i)
    cv2.waitKey(0)
    if img_right is None:
        print("Failed to load image: ", image_dir + "right_%02d.png" % i)
        continue

    # konwersja do odcieni szarości
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # znalezienie narożników szachownicy dla obrazów lewego i prawego
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, height),
                                                      cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                      cv2.CALIB_CB_FAST_CHECK +
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height),
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                        cv2.CALIB_CB_FAST_CHECK +
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)

    # pominięcie obrazu jeśli wierzchołki szachownicy są zbyt blisko krawędzi obrazu
    if not ret_left or not ret_right:
        print("Chessboard not detected in either left or right image. Image pair: ", i)
        continue

    # zwiekszenie dokładności znalezionych narożników
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria)

    # jeśli znaleziono, dodaj punkty obiektu, punkty obrazu (po zwiększeniu dokładności)
    objpoints.append(objp)  # użyj tego samego obiektu 3D dla wszystkich obrazów
    imgpointsLeft.append(corners_left)
    imgpointsRight.append(corners_right)

disparity_bm_orig = stereo_bm.compute(gray_left, gray_right)
disparity_sgbm_orig = stereo_sgbm.compute(gray_left, gray_right)
# Normalize the disparity maps
disparity_bm_normalized_orig = cv2.normalize(disparity_bm_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disparity_sgbm_normalized_orig = cv2.normalize(disparity_sgbm_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# konwersja do macierzy numpy
objpoints = np.asarray(objpoints)
imgpointsLeft = np.asarray(imgpointsLeft)
imgpointsRight = np.asarray(imgpointsRight)

# KALIBRACJA
# zmienne dla obrazu lewego
# N_OK - liczba obrazów, na których znaleziono szachownicę
N_OK = len(objpoints)
# macierz do przechowywania macierzy kamery i współczynników zniekształceń
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
# macierz do przechowywania wektorów rotacji i translacji
rvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_left = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# kalibracja dla obrazu lewego
ret, K_left, D_left, _, _ =  cv2.fisheye.calibrate(
    objpoints,
    imgpointsLeft,
    image_size,
    K_left,
    D_left,
    rvecs_left,
    tvecs_left,
    calibration_flags,
    criteria
)

# zmienne dla obrazu prawego
N_OK = len(objpoints)
K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_right = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# kalibracja dla obrazu prawego
ret, K_right, D_right, _, _ =  cv2.fisheye.calibrate(
    objpoints,
    imgpointsRight,
    image_size,
    K_right,
    D_right,
    rvecs_right,
    tvecs_right,
    calibration_flags,
    criteria
)

# wyświetlenie wyników
print("Camera parameters for the left image:")
print("Camera matrix (K_left):\n", K_left)
print("Distortion coefficients (D_left):\n", D_left)

print("Camera parameters for the right image:")
print("Camera matrix (K_right):\n", K_right)
print("Distortion coefficients (D_right):\n", D_right)

# kalibracja stereo
(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
        objpoints, imgpointsLeft, imgpointsRight,
        K_left, D_left,
        K_right, D_right,
        image_size, None, None,
        cv2.CALIB_FIX_INTRINSIC,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
)

print("Stereo Calibration RMS:", RMS)
print("Rotation Matrix (R):\n", rotationMatrix)
print("Translation Vector (T):\n", translationVector)

# Rektyfikacja obrazu
# macierz rotacji dla układu po rektyfikacji
R2 = np.zeros([3, 3])
# macierze projekcji dla układu po rektyfikacji
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
# macierz mapy dysparycji używana do przeliczania disparity na rzeczywiste odległości w jednostkach przestrzeni trójwymiarowej
Q = np.zeros([4, 4])

(leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepthMap) = \
    cv2.fisheye.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        image_size,
        rotationMatrix, translationVector,
        0, R2, P1, P2, Q,
        cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0
    )

# inicializacja mapy dysparycji do rektyfikacji obrazu
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, leftRectification,
    leftProjection, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, rightRectification,
    rightProjection, image_size, cv2.CV_16SC2)

# wczytanie obrazów
img_left = cv2.imread(image_dir + "left_01.png")
img_right = cv2.imread(image_dir + "right_01.png")

# rektyfikacja obrazów
dst_L = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1] # RGB image size
visRectify = np.zeros((YY, XX*2, N), np.uint8) # create a new image with anew size (height, 2*width)
visRectify[:,0:XX:,:] = dst_L # left image assignment
visRectify[:,XX:XX*2:,:] = dst_R # right image assignment

# narysowanie linii poziomych na obrazach
for y in range(0,YY,10):
    cv2.line(visRectify, (0,y), (XX*2,y), (255,0,0))
cv2.imshow('visRectify',visRectify) # wyświetlenie obrazu z liniami 
cv2.waitKey(0)
cv2.destroyAllWindows()

# wczytanie obrazów
img_left = cv2.imread(image_dir + "left_05.png")
img_right = cv2.imread(image_dir + "right_05.png")

# rektyfikacja obrazów
dst_L = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1] # RGB image size
visRectify = np.zeros((YY, XX*2, N), np.uint8) # create a new image with anew size (height, 2*width)
visRectify[:,0:XX:,:] = dst_L # left image assignment
visRectify[:,XX:XX*2:,:] = dst_R # right image assignment

# narysowanie linii poziomych na obrazach
for y in range(0,YY,10):
    cv2.line(visRectify, (0,y), (XX*2,y), (255,0,0))
cv2.imshow('visRectify',visRectify) # wyświetlenie obrazu z liniami 
cv2.waitKey(0)
cv2.destroyAllWindows()

# wczytanie obrazów
img_left = cv2.imread(image_dir + "left_11.png")
img_right = cv2.imread(image_dir + "right_11.png")

# rektyfikacja obrazów
dst_L = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1] # RGB image size
visRectify = np.zeros((YY, XX*2, N), np.uint8) # create a new image with anew size (height, 2*width)
visRectify[:,0:XX:,:] = dst_L # left image assignment
visRectify[:,XX:XX*2:,:] = dst_R # right image assignment

# narysowanie linii poziomych na obrazach
for y in range(0,YY,10):
    cv2.line(visRectify, (0,y), (XX*2,y), (255,0,0))
cv2.imshow('visRectify',visRectify) # wyświetlenie obrazu z liniami 
cv2.waitKey(0)
cv2.destroyAllWindows()

disparity_bm = stereo_bm.compute(gray_left, gray_right)
disparity_sgbm = stereo_sgbm.compute(gray_left, gray_right)

# Normalize the disparity maps
disparity_bm_normalized = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disparity_sgbm_normalized = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display original images
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
plt.title("Original Left Image")
plt.subplot(4, 2, 2)
plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
plt.title("Original Right Image")

# Display rectified and undistorted images
plt.subplot(4, 2, 3)
plt.imshow(cv2.cvtColor(dst_L, cv2.COLOR_BGR2RGB))
plt.title("Rectified and Undistorted Left Image")
plt.subplot(4, 2, 4)
plt.imshow(cv2.cvtColor(dst_R, cv2.COLOR_BGR2RGB))
plt.title("Rectified and Undistorted Right Image")

# Display disparity maps
plt.subplot(4, 2, 5)
plt.imshow(disparity_bm_normalized, cmap="gray")
plt.title("Disparity Map BM calib")
plt.subplot(4, 2, 6)
plt.imshow(disparity_sgbm_normalized, cmap="gray")
plt.title("Disparity Map SGM calib")

plt.subplot(4, 2, 7)
plt.imshow(disparity_bm_normalized_orig, cmap="gray")
plt.title("Disparity Map BM no calib")
plt.subplot(4, 2, 8)
plt.imshow(disparity_sgbm_normalized_orig, cmap="gray")
plt.title("Disparity Map SGM no calib")

plt.tight_layout()
plt.show()
