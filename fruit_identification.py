import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog

import glob

def FtrExtractHOG(img):
    #Preprocessing using grayscale and resize
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=resize(img, (72, 72),anti_aliasing=True)
    #Feature Extraction using HOG
    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, multichannel=False)
    return ftr

def FtrExtractColorHist(img):
    chans = cv2.split(img)
    colors = ("h", "s", "v")
    features = []

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
    return np.array(features).flatten()

def loadimage(arr,n,name_of_fruit):
    label=[]
    flag=0
    for i in range(n):
        strr = "Image/"+name_of_fruit+"_"+str(i+1)+"/*.jpg" #make list from name of dirctory
        #print(strr)
        for file in glob.glob(strr):
            img=np.asarray(plt.imread(file))
            arr.append(img)
            label.append(name_of_fruit)
    return arr,label

apple=[]
banana =[]
lemon=[]
lime=[]
orange=[]
peach=[]
pear=[]
carambula=[]
strawberry=[]

apple,label_apple=loadimage(apple,5,"apple")
banana,label_banana=loadimage(banana,4,"banana")
lemon,label_lemon=loadimage(lemon,6,"lemon")
lime,label_lime=loadimage(lime,1,"limes")
orange,label_orange=loadimage(orange,4,"orange")
peach,label_peach=loadimage(peach,3,"peach")
pear,label_pear=loadimage(pear,3,"pear")
carambula,label_carambula=loadimage(carambula, 1, "carambula")
strawberry,label_strawberry=loadimage(strawberry, 2, "strawberry")

raw_atribut = {'Kelas': ['Apple','Banana','Lemon','Lime','Orange', 'Pear', 'Peach','carambula','strawberry'],
           'Jumlah': [np.shape(apple)[0],np.shape(banana)[0],np.shape(lemon)[0],np.shape(lime)[0],np.shape(orange)[0],np.shape(peach)[0],np.shape(pear)[0],np.shape(carambula),np.shape(strawberry)]}
atribut= pd.DataFrame(raw_atribut,
                       columns=['Kelas','Jumlah'])
atribut


print('Example of the Dataset') #show the data set
fig = plt.figure()
ax1 = fig.add_subplot(3,3,1)
ax1.set_title('Apple')
ax1.set_axis_off()
ax1.imshow(apple[0])

ax2 = fig.add_subplot(3,3,2)
ax2.set_title('Banana')
ax2.set_axis_off()
ax2.imshow(banana[0])

ax3 = fig.add_subplot(3,3,3)
ax3.set_title('Lemon')
ax3.set_axis_off()
ax3.imshow(lemon[0])

ax4 = fig.add_subplot(3,3,4)
ax4.set_title('Lime')
ax4.set_axis_off()
ax4.imshow(lime[0])

ax5 = fig.add_subplot(3,3,5)
ax5.set_title('Orange')
ax5.set_axis_off()
ax5.imshow(orange[0])

ax6 = fig.add_subplot(3,3,6)
ax6.set_title('Pear')
ax6.set_axis_off()
ax6.imshow(pear[0])

ax7 = fig.add_subplot(3,3,7)
ax7.set_title('Peach')
ax7.set_axis_off()
ax7.imshow(peach[0])

ax8 = fig.add_subplot(3,3,8)
ax8.set_title('carambula')
ax8.set_axis_off()
ax8.imshow(carambula[0])

ax9 = fig.add_subplot(3,3,9)
ax9.set_title('strawberry')
ax9.set_axis_off()
ax9.imshow(strawberry[0])

def preprocessing1(arr): #Preprocessing of the image,change the image to grey and resize an image
    arr_prep=[]
    for i in range(np.shape(arr)[0]):
        img=cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img=resize(img, (72, 72),anti_aliasing=True)
        arr_prep.append(img)
    return arr_prep

def featureExtraction1(arr): #Extract an attribute 
    arr_feature=[]
    for i in range(np.shape(arr)[0]):
        arr_feature.append(FtrExtractHOG(arr[i]))
    return arr_feature
#Connect the data from each class
X_Shapedes =np.concatenate((apple,banana,lemon,lime,orange,peach,pear,carambula,strawberry))
y_Shapedes =np.concatenate((label_apple,label_banana,label_lemon,label_lime,label_orange,label_peach,label_pear,label_carambula,label_strawberry))

X_train, X_test, y_train, y_test = train_test_split(X_Shapedes, y_Shapedes, test_size=0.33, random_state=42)

print('Num of Data Train :',X_train.shape[0])
print('Num of Data Test  :',X_test.shape[0])

X_trainp=preprocessing1(X_train)
X_testp=preprocessing1(X_test)

X_trainftr=featureExtraction1(X_trainp)
X_testftr=featureExtraction1(X_testp)

#Make the classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=11)
knn_clf.fit(X_trainftr, y_train)

y_knn_pred = knn_clf.predict(X_testftr)

print(accuracy_score(y_test, y_knn_pred)*100,'%')

slice = 17

plt.figure(figsize=(16,8))
for i in range(slice):
    plt.subplot(1, slice, i+1)
    plt.imshow(X_test[i], interpolation='nearest')
    plt.text(0, 0, y_knn_pred[i], color='black', 
             bbox=dict(facecolor='white', alpha=1))
    plt.axis('off')
