import numpy as np
from skimage.feature import local_binary_pattern as lbp
import cv2
face_cascade = cv2.CascadeClassifier('/home/vakidzaci/cv/.env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def LBP(image, points=8, radius=1):
    '''
    Uniform Local Binary Patterns algorithm
    Input image with shape (height, width, channels)
    Output features with length * number of channels
    '''
    # calculate pattern length
    length = points**2 - abs(points - 3)
    # lbp per channel in image
    histogram = np.empty(0, dtype=np.int)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        pattern = lbp(channel, points, radius, method='nri_uniform')
        pattern = pattern.astype(np.int).ravel()
        pattern = np.bincount(pattern)
        if len(pattern) < length:
            pattern = np.concatenate((pattern, np.zeros(59 - len(pattern))))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    features = (histogram - np.mean(histogram)) / np.std(histogram)
    return features

def CoALBP(image, lbp_r=1, co_r=2):
    '''
    Co-occurrence of Adjacent Local Binary Patterns algorithm
    Input image with shape (height, width, channels)
    Input lbp_r is radius for adjacent local binary patterns
    Input co_r is radius for co-occurence of the patterns
    Output features with length 1024 * number of channels
    '''
    h, w, c = image.shape
    # normalize face
    image = (image - np.mean(image, axis=(0,1))) / (np.std(image, axis=(0,1)) + 1e-8)
    # albp and co-occurrence per channel in image
    histogram = np.empty(0, dtype=np.int)
    for i in range(image.shape[2]):
        C = image[lbp_r:h-lbp_r, lbp_r:w-lbp_r, i].astype(float)
        X = np.zeros((4, h-2*lbp_r, w-2*lbp_r))
        # adjacent local binary patterns
        X[0, :, :] = image[lbp_r  :h-lbp_r  , lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
        X[1, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r  :w-lbp_r  , i] - C
        X[2, :, :] = image[lbp_r  :h-lbp_r  , lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
        X[3, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r  :w-lbp_r  , i] - C
        X = (X>0).reshape(4, -1)
        # co-occurrence of the patterns
        A = np.dot(np.array([1, 2, 4, 8]), X)
        A = A.reshape(h-2*lbp_r, w-2*lbp_r) + 1
        hh, ww = A.shape
        D  = (A[co_r  :hh-co_r  , co_r  :ww-co_r  ] - 1) * 16 - 1
        Y1 =  A[co_r  :hh-co_r,   co_r+co_r:ww-co_r+co_r] + D
        Y2 =  A[co_r-co_r:hh-co_r-co_r, co_r+co_r:ww-co_r+co_r] + D
        Y3 =  A[co_r-co_r:hh-co_r-co_r, co_r  :ww-co_r  ] + D
        Y4 =  A[co_r-co_r:hh-co_r-co_r, co_r-co_r:ww-co_r-co_r] + D
        Y1 = np.bincount(Y1.ravel(), minlength=256)
        Y2 = np.bincount(Y2.ravel(), minlength=256)
        Y3 = np.bincount(Y3.ravel(), minlength=256)
        Y4 = np.bincount(Y4.ravel(), minlength=256)
        pattern = np.concatenate((Y1, Y2, Y3, Y4))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    features = (histogram - np.mean(histogram)) / np.std(histogram)
    return histogram

def gethist(path,ch):
    img = cv2.imread(path + ch)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        histogram = np.array([])
        x,y,w,h = faces[0]
        crop_face = img[y:y+w, x:x+h]
        crop_face = cv2.resize(crop_face, (64, 64))
        # imgYCC = cv2.cvtColor(crop_face, cv2.COLOR_BGR2HSV)
        radius = 2
        n_points = 16
        # for i in range(imgYCC.shape[2]):
        #     lbp = local_binary_pattern(imgYCC[:,:,i], n_points, radius, method='uniform')
        #     histogram = np.append(histogram,lbp.ravel())
        return CoALBP(crop_face)

def preproc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        histogram = np.array([])
        x,y,w,h = faces[0]
        crop_face = img[y:y+w, x:x+h]
        crop_face = cv2.resize(crop_face, (64, 64))
        # imgYCC = cv2.cvtColor(crop_face, cv2.COLOR_BGR2HSV)
        radius = 2
        n_points = 16
        # for i in range(imgYCC.shape[2]):
        #     lbp = local_binary_pattern(imgYCC[:,:,i], n_points, radius, method='uniform')
        #     histogram = np.append(histogram,lbp.ravel())
        return CoALBP(crop_face),faces

def gethistHSV(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        histogram = np.array([])
        x,y,w,h = faces[0]
        crop_face = gray[y:y+w, x:x+h]
        crop_face = cv2.resize(crop_face, (64, 64))
        radius = 2
        n_points = 16
        # for i in range(crop_face.shape[2]):
        l = lbp(crop_face, n_points, radius, method='uniform')
        histogram = np.append(histogram,l.ravel())
        return histogram,faces
