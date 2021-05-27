import numpy as np
import cv2 
import glob
from google.colab.patches import cv2_imshow
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images=  glob.glob('drive/MyDrive/Temp/test/*')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        cv2_imshow(img)
        cv2.waitKey(500)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img=  cv2.imread('drive/MyDrive/Temp/test/IMG_test.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print(mtx)
print(rvecs)
print(tvecs)
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
tot_error=0
total_points=0
for i in range(len(objpoints)):
    reprojected_points, _=cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    reprojected_points=reprojected_points.reshape(-1,2)
    tot_error+=np.sum(np.abs(imgpoints [i]-reprojected_points)**2)
    total_points+=len(objpoints[i])

mean_error=np.sqrt(tot_error/total_points)
print("Mean reprojection error:", mean_error)
