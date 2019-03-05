import numpy as np
import cv2
from skimage.feature import local_binary_pattern,multiblock_lbp
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.svm import LinearSVC,SVC
import sys
sys.path.insert(0, '/home/vakidzaci/fr/color_texture_analysis/methods')
from methods import CoALBP
from methods import gethist
import time





# img = cv2.imread('madi.jpg')
# img = cv2.resize(img, (128, 128))
# imgYCC = cv2.cv2tColor(img, cv2.COLOR_BGR2YCR_CB)


C = '/home/vakidzaci/cv/data/ClientRaw/'
I = '/home/vakidzaci/cv/data/ImposterRaw/'
face_cascade = cv2.CascadeClassifier('/home/vakidzaci/cv/.env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def train():
    q = 10
    imposter_train = pd.read_csv('/home/vakidzaci/cv/data/imposter_train_raw.csv')
    client_train = pd.read_csv('/home/vakidzaci/cv/data/client_train_raw.csv')

    # imposter_train = imposter_train.head(q)
    # client_train = client_train.head(q)

    imposter_train['fraud'] = 1
    client_train['fraud'] = 0


    train = pd.concat([imposter_train, client_train])




    H = []
    y_train = []
    for index, row in train.iterrows():
        if row['fraud'] == 0:
            c = gethist(C,row['path'])
            if c is not None:
                H.append(c)
                y_train.append(0)
        else:
            i = gethist(I,row['path'])
            if i is not None:
                H.append(i)
                y_train.append(1)


    H = np.asarray(H)
    y_train = np.asarray(y_train)



    print(H.shape)
    print(y_train.shape)

    clf = SVC(probability=True)
    clf.fit(H,y_train)
    dump(clf, '/home/vakidzaci/fr/color_texture_analysis/models/SVCspoofYCrCb.joblib')

def test():
    print("test")

if __name__ == "__main__":
    start = time.time()

    train()

    end = time.time()
    print(end - start)
