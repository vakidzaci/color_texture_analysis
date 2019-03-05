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
from methods import preproc,gethistHSV


clf = load('/home/vakidzaci/fr/color_texture_analysis/models/xgboost.joblib')
cap = cv2.VideoCapture(1)

if cap is None or not cap.isOpened():
    raise NameError('CAMERA IS NOT FOUND')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # print(frame)
    prep = gethistHSV(frame)
    print(frame)
    if prep is not None:
        feature_vector,faces = prep
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        prediction = clf.predict_proba(feature_vector)
        point = (x, y-5)
        if prediction[0][1] >= 0.8:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=frame, text="False", org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                        thickness=2, lineType=cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=frame, text="True", org=point, fontFace=font, fontScale=0.9, color=(0, 255, 0),
                        thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
