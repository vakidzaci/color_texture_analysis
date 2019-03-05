import numpy as np
import cv2
from skimage.feature import local_binary_pattern,multiblock_lbp
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score

import time
start = time.time()

# img = cv2.imread('madi.jpg')
# img = cv2.resize(img, (128, 128))
# imgYCC = cv2.cv2tColor(img, cv2.COLOR_BGR2YCR_CB)


C = '/home/vakidzaci/cv/data/ClientRaw/'
I = '/home/vakidzaci/cv/data/ImposterRaw/'
face_cascade = cv2.CascadeClassifier('/home/vakidzaci/cv/.env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def gethist(path,ch):
    img = cv2.imread(path + ch)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        histogram = np.array([])
        x,y,w,h = faces[0]
        crop_face = img[y:y+w, x:x+h]
        crop_face = cv2.resize(crop_face, (64, 64))
        imgYCC = cv2.cvtColor(crop_face, cv2.COLOR_BGR2YCR_CB)

        radius = 2
        n_points = 16
        for i in range(imgYCC.shape[2]):
            lbp = local_binary_pattern(imgYCC[:,:,i], n_points, radius, method='uniform')
            histogram = np.append(histogram,lbp.ravel())
        return histogram
if __name__ == "__main__":
    q = 10
    imposter_train = pd.read_csv('/home/vakidzaci/cv/data/imposter_train_raw.csv').head(q)
    imposter_test = pd.read_csv('/home/vakidzaci/cv/data/imposter_test_raw.csv').head(q)
    client_train = pd.read_csv('/home/vakidzaci/cv/data/client_train_raw.csv').head(q)
    client_test = pd.read_csv('/home/vakidzaci/cv/data/client_test_raw.csv').head(q)

    imposter_train['fraud'] = 1
    imposter_test['fraud'] = 1
    client_train['fraud'] = 0
    client_test['fraud'] = 0


    train = pd.concat([imposter_train, client_train])
    test = pd.concat([imposter_test,client_test])
    P = train[['path']]
    Y = train[['fraud']]



    # H = []
    y_test = []
    y_predict = []
    for index, row in test.iterrows():
        if row['fraud'] == 0:
            c = gethist(C,row['path'])
            if c is not None:
                H.append(c)
                y_test.append(0)
        else:
            i = gethist(I,row['path'])
            if i is not None:
                H.append(i)
                y_test.append(1)

    H = np.asarray(H)
    y_test = np.asarray(y_test)


    #
    # print(H.shape)
    # print(y_test.shape)
    #
    # clf = load('/home/vakidzaci/fr/color_texture_analysis/models/SVCloop.joblib')

    y_predict = clf.predict_proba(H)
    print(y_predict)
    # y_predict = clf.predict(H)
    # print(accuracy_score(y_test,y_predict))
    #
    # end = time.time()
    # print(end - start)
