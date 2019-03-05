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
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from methods import gethist

import time
start = time.time()

# img = cv2.imread('madi.jpg')
# img = cv2.resize(img, (128, 128))
# imgYCC = cv2.cv2tColor(img, cv2.COLOR_BGR2YCR_CB)


C = '/home/vakidzaci/cv/data/ClientRaw/'
I = '/home/vakidzaci/cv/data/ImposterRaw/'


if __name__ == "__main__":
    q = 100
    imposter_train = pd.read_csv('/home/vakidzaci/cv/data/imposter_train_raw.csv')
    imposter_test = pd.read_csv('/home/vakidzaci/cv/data/imposter_test_raw.csv')
    client_train = pd.read_csv('/home/vakidzaci/cv/data/client_train_raw.csv')
    client_test = pd.read_csv('/home/vakidzaci/cv/data/client_test_raw.csv')

    imposter_train = imposter_train.head(q)
    imposter_test = imposter_test.head(q)
    client_train = client_train.head(q)
    client_test = client_test.head(q)

    imposter_train['fraud'] = 1
    imposter_test['fraud'] = 1
    client_train['fraud'] = 0
    client_test['fraud'] = 0

    # print(client_test.shape)
    # print(imposter_test.shape)

    train = pd.concat([imposter_train, client_train])
    test = pd.concat([imposter_test,client_test])
    print(test)
    # P = train[['path']]
    # Y = train[['fraud']]



    H = []
    y_test = []
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



    print(H.shape)
    print(y_test.shape)

    clf = load('/home/vakidzaci/fr/color_texture_analysis/models/SVCspoofYCrCb.joblib')

    y_predict = clf.predict_proba(H)
    print(y_predict)
    # print(y_predict)
    # print("Accuracy Score : {}".format(accuracy_score(y_test,y_predict)))
    # print("Recall Score : {}".format(recall_score(y_test,y_predict)))
    # print("Precision Score : {}".format(precision_score(y_test,y_predict)))
    #
    # end = time.time()
    # print(end - start)
