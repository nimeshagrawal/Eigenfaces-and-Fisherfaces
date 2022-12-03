#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import math


# In[2]:



#Image read
img = cv2.imread('1_1.tif', 0)
print(img.shape)

img = cv2.resize(img, (100, 120))
print(img.shape)
cv2.imshow('image', img)
key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows()

cv2.destroyAllWindows()


# In[ ]:





# In[17]:


#Data For Two class

X_data = np.append(cv_img_4, cv_img_9, axis = 0)
Y_data = np.append(Y_4, Y_9, axis = 0)
X_data.shape


# In[39]:


#Data For multiple class

X_data = np.append(cv_img_1, cv_img_2, axis = 0)
#X_data = np.append(X_data, cv_img_3, axis = 0)
#X_data = np.append(X_data, cv_img_4, axis = 0)
#X_data = np.append(cv_img_4, cv_img_5, axis = 0)
X_data = np.append(X_data, cv_img_5, axis = 0)
X_data = np.append(X_data, cv_img_6, axis = 0)
X_data = np.append(X_data, cv_img_7, axis = 0)
X_data = np.append(X_data, cv_img_8, axis = 0)
#X_data = np.append(X_data, cv_img_9, axis = 0)
#X_data = np.append(X_data, cv_img_10, axis = 0)
#X_data = np.append(X_data, cv_img_11, axis = 0)
#X_data = np.append(X_data, cv_img_12, axis = 0)

Y_data = np.append(Y_1, Y_2, axis = 0)
#Y_data = np.append(Y_data, Y_3, axis = 0)
#Y_data = np.append(Y_data, Y_4, axis = 0)
Y_data = np.append(Y_data, Y_5, axis = 0)
#Y_data = np.append(Y_4, Y_5, axis = 0)
Y_data = np.append(Y_data, Y_6, axis = 0)
Y_data = np.append(Y_data, Y_7, axis = 0)
Y_data = np.append(Y_data, Y_8, axis = 0)
#Y_data = np.append(Y_data, Y_9, axis = 0)
#Y_data = np.append(Y_data, Y_10, axis = 0)
#Y_data = np.append(Y_data, Y_11, axis = 0)
#Y_data = np.append(Y_data, Y_12, axis = 0)

X_data.shape


# In[59]:


def PCA_(X_data, Y_data):

    X = X_data.T
    XXT = (X.T).dot(X)

    eigen_values, eigen_vectors = np.linalg.eig(XXT)
    lembda = np.count_nonzero(eigen_values, axis=0)

    #Energy in eigenvalue
    e = abs(eigen_values)
    e = -np.sort(-e)
    sum_egn = sum(e)#Sum of total eigenvalues
    a = 0
    count = 0
    for i in range (30):
        a += e[i]    
        count = count+1
        if a/sum_egn >= 0.95:
            break

    #print(count)

    #print("Percentage energy", a/sum_egn)

    #Stacking in significant eigen values
    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)


    # Stacking wigenvalues and eigenvectors in decreasing order of significance 
    Sig_Eig = []
    n = lembda

    if n == 1:
        Sig_Eig = np.hstack((pairs[0][1].reshape(len(eigen_values),1))).real.reshape(-1,1)

    else:
        for i in range(n):
            W1 = np.hstack((pairs[i][1].reshape(len(eigen_values),1))).real.reshape(-1,1)
            #print(W1)
            Sig_Eig.append(W1)

    Sig_Eig = np.array(Sig_Eig).reshape(n,len(eigen_values)).T

    U = np.zeros((X.shape[0], 1))
    for i in range (lembda):
        k = Sig_Eig[i].reshape(-1,1)
        L = ((X).dot(k))/(math.sqrt(pairs[i][0]))
        if i == 0:
            U = L
        else:
            U = np.append(U, L, axis =1)

    tot_cls = np.count_nonzero(np.unique(Y_data, return_counts=False))
    if tot_cls == 2:
        c = 2
    else:
        c = tot_cls*5

    U = U[:, 0:c]

    X_pca = (X_data).dot(U)
    #print(X_pca.shape)
    return(X_pca)


# In[19]:


def Sb_Sw(X_train, Y_train):    
    
    inputs = X_train
    targets = Y_train
    ####Finding unique classes######
    unq = np.unique(targets, return_counts=False)
    #print(unq)


    #####Separating data class wise
    input_class = []
    target_class = []
    for i in range (unq.shape[0]):
        mask = targets == unq[i]
        input_class.append(inputs[mask])

    input_class = np.array(input_class)
    print(input_class[0].shape)


    ######Class wise mean
    input_class_mean = []
    for i in range (input_class.shape[0]):
        input_class_mean.append(np.mean(input_class[i], axis = 0))

    input_class_mean = np.array(input_class_mean)

    ######Overall mean
    input_mean = np.mean(inputs, axis = 0)
    #print(input_class_mean.shape)



    ######## SB Calculation

    if (input_class_mean.shape[0] == 2):
        B = input_class_mean[0] - input_class_mean[1]
        Sb = ((B[:,None]).dot(B[None,:]))

    else:
        Sb = np.zeros((input_class_mean.shape[1], input_class_mean.shape[1]))
        for i in range (input_class_mean.shape[0]):
            n = input_class[i].shape[0]
            B1 = input_class_mean[i] - input_mean
            Sb += n*((B1[:,None]).dot(B1[None,:]))
    #print(Sb.shape)



    ######## SW Calculation

    Sw = np.zeros((input_class_mean.shape[1], input_class_mean.shape[1]))

    for i in range (input_class_mean.shape[0]):
        n = input_class[i].shape[0]
        W1 = input_class[i] - input_class_mean[i]
        Sw += ((W1.T).dot(W1))

    #print(Sw.shape)
    
    
    return (Sb, Sw, input_class_mean)


# In[20]:


def eig(Sb, Sw):    
    
    ######## Calculating Sw-1*Sb & Eigenvalues 

    Sw_inv = np.linalg.inv(Sw)

    Sw_inv_Sb = (Sw_inv).dot(Sb)

    eigen_values, eigen_vectors = np.linalg.eig(Sw_inv_Sb)
    
    return (eigen_values, eigen_vectors)


# In[21]:


def plot(X_fda, Y_train):    
    from sklearn.decomposition import PCA
    
    ####### Projected data visulization of Fisher Plane/Line

    Y = Y_train
    if X_fda.shape[1] ==1:
        plt.scatter(X_fda,Y, c=Y,cmap='rainbow',alpha=0.7,edgecolors='g')
        plt.show()
    elif X_fda.shape[1] ==2:
        plt.scatter(X_fda[:,0],X_fda[:,1], c=Y,cmap='rainbow',alpha=0.7,edgecolors='g')
        plt.show()
    else:
        pca = PCA(n_components=2)
        X_mn_pca = pca.fit_transform(X_fda)
        plt.scatter(X_mn_pca[:,0], X_mn_pca[:,1], c=Y,cmap='rainbow',alpha=0.7,edgecolors='g')
        plt.show()


# In[22]:


def graph(tot_tr_Acc, tot_te_Acc, n_component):    
    
    from scipy.interpolate import interp1d

    y_train = tot_tr_Acc
    y_test = tot_te_Acc

    x = np.arange(1, n_component+1)

    cubic_interploation_model = interp1d(x, y_train, kind = "linear")

    # Plotting the Graph
    X_train=np.linspace(x.min(), x.max(), 50)
    Y_train=cubic_interploation_model(X_train)

    plt.plot(X_train,Y_train) 


    plt.title("Training accuracy")
    plt.xlabel("Number of Eigenvector") 
    plt.ylabel("Accuracy") 
    plt.grid()
    plt.show()


    cubic_interploation_model = interp1d(x, y_test, kind = "linear")

    # Plotting the Graph
    X_test=np.linspace(x.min(), x.max(), 50)
    Y_test=cubic_interploation_model(X_test)

    plt.plot(X_test,Y_test) 


    plt.title("Test accuracy")
    plt.xlabel("Number of Eigenvector") 
    plt.ylabel("Accuracy") 
    plt.grid()
    plt.show()


# In[25]:


def classification(X_train_fda, Y_train, X_test_fda, Y_test):


    Y = Y_train
    X_fda = X_train_fda
    unq = np.unique(Y, return_counts=False)

    X_cl = []

    for i in range (unq.shape[0]):
        mask = Y == unq[i]
        X_cl.append(X_fda[mask])

    X_cl = np.array(X_cl)


    mean = []
    for i in range (X_cl.shape[0]):
        mean.append(np.mean(X_cl[i], axis =0))

    mean = np.array(mean)

    #Train data prediction

    Y_train_pred = []
    for i in range (X_fda.shape[0]):
        distance=euclidean_distances(mean, X_fda[i].reshape(1,-1))
        a = np.argmin(distance)
        Y_train_pred.append(unq[a])

    Y_train_pred = np.array(Y_train_pred)
    #print("\n-------- Train : Actual Label -------")
    #print(Y)
    #print("\n------- Train : Predictaed Label -------")
    #print(Y_train_pred)



    #Test data prediction

    Y_test_pred = []
    for i in range (X_test_fda.shape[0]):
        distance=euclidean_distances(mean, X_test_fda[i].reshape(1,-1))
        a = np.argmin(distance)
        Y_test_pred.append(unq[a])

    Y_test_pred = np.array(Y_test_pred)

    print("\n---------- Test : Actual Label --------")
    print(Y_test)
    print("\n---------- Test : Predictaed Label -------")
    print(Y_test_pred)

    Train_Accuracy = (accuracy_score(Y, Y_train_pred))
    Test_Accuracy = (accuracy_score(Y_test, Y_test_pred))
                          
    print('\nTrain Accuracy : ' + str(accuracy_score(Y, Y_train_pred)))
    print('Test Accuracy : ' + str(accuracy_score(Y_test, Y_test_pred)))

    return(Train_Accuracy, Test_Accuracy)


# In[24]:


def X_fda(eigen_values, eigen_vectors, input_class_mean, n_component, X_train, Y_train, X_test, Y_test):    
    
    #Making eigne value eigenvector pair and printing eigenvalues
    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    ######## Calculating Fisher Hyperplane/Line

    
    n = n_component
    tot_tr_Acc = np.zeros(n)
    tot_te_Acc = np.zeros(n)
    for i in range (n):
        W = []
        if n == 1:
            W = np.hstack((pairs[0][1].reshape(input_class_mean.shape[1],1))).real.reshape(-1,1)
            
        else:
            for j in range(i+1):
                W1 = np.hstack((pairs[j][1].reshape(input_class_mean.shape[1],1))).real.reshape(-1,1)
                W.append(W1)

        W = np.array(W).reshape((i+1),input_class_mean.shape[1]).T

        #print("input shape:", inputs.shape)
        print("W shape:    ", W.shape)

        X_train_fda = (X_train.dot(W))
        X_test_fda = (X_test.dot(W))
        
        print("\n \nNumber of Eigen Vectors:", i+1)
        
        if i == n-1:   
            plot(X_train_fda, Y_train)
        
        Train_Accuracy, Test_Accuracy = classification(X_train_fda, Y_train, X_test_fda, Y_test)

        
        tot_tr_Acc[i] = Train_Accuracy
        tot_te_Acc[i] = Test_Accuracy
    
    return(tot_tr_Acc, tot_te_Acc)


# In[60]:


#Implementation

X_data = X_data
Y_data = Y_data 

X_pca = PCA_(X_data, Y_data)

inputs, X_test, targets, Y_test = train_test_split(X_pca, Y_data, test_size=0.20)

#img_show(X_data)
#img_show(X_data[1].reshape(120, 100))

#program with train data 
Sb, Sw, input_class_mean = Sb_Sw(inputs, targets)

eigen_values, eigen_vectors = eig(Sb, Sw)

n_component = 7

X_train = inputs
Y_train = targets
tot_tr_Acc, tot_te_Acc = X_fda(eigen_values, eigen_vectors, input_class_mean, n_component, X_train, Y_train, X_test, Y_test)

if n_component !=1:
    graph (tot_tr_Acc, tot_te_Acc, n_component)


# In[ ]:





# In[ ]:





# In[ ]:


#LOADING IMAGES


# In[31]:


# Imgae Class 1
import glob
import cv2 as cv

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/senthil_database_version1/senthil_database_version1/S1/*.tif")
cv_img_1 = []
for img in path:
    n = cv.imread(img, 0)
    n = cv2.resize(n, (100, 120))
    cv_img_1.append(n)
cv_img_1 = np.array(cv_img_1)
print(cv_img_1.shape)
cv_img_1 = cv_img_1.reshape(16, -1)
print(cv_img_1.shape)


Y_1 = np.ones(cv_img_1.shape[0])

print(Y_1)

# Imgae Class 2
path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/senthil_database_version1/senthil_database_version1/S2/*.tif")
cv_img_2 = []
for img in path:
    n = cv.imread(img, 0)
    n = cv2.resize(n, (100, 120))
    cv_img_2.append(n)
cv_img_2 = np.array(cv_img_2)
print(cv_img_2.shape)
cv_img_2 = cv_img_2.reshape(16, -1)
print(cv_img_2.shape)

Y_2 = np.ones(cv_img_2.shape[0])
Y_2 = Y_2*(2)
print(Y_2)

# Imgae Class 3
path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/senthil_database_version1/senthil_database_version1/S4/*.tif")
cv_img_3 = []
for img in path:
    n = cv.imread(img, 0)
    n = cv2.resize(n, (100, 120))
    cv_img_3.append(n)
cv_img_3 = np.array(cv_img_3)
print(cv_img_3.shape)
cv_img_3 = cv_img_3.reshape(16, -1)
print(cv_img_3.shape)

Y_3 = np.ones(cv_img_3.shape[0])
Y_3 = Y_3*(3)
print(Y_3)

# Imgae Class 4
path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/senthil_database_version1/senthil_database_version1/S5/*.tif")
cv_img_4 = []
for img in path:
    n = cv.imread(img, 0)
    n = cv2.resize(n, (100, 120))
    cv_img_4.append(n)
cv_img_4 = np.array(cv_img_4)
print(cv_img_4.shape)
cv_img_4 = cv_img_4.reshape(16, -1)
print(cv_img_4.shape)

Y_4 = np.ones(cv_img_4.shape[0])
Y_4 = Y_4*(4)
print(Y_4)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S5/*.gif")
cv_img_5 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (100, 120))
    cv_img_5.append(first_gray)
    
cv_img_5 = np.array(cv_img_5)
print(cv_img_5.shape)
cv_img_5 = cv_img_5.reshape(cv_img_5.shape[0], -1)
print(cv_img_5.shape)

Y_5 = np.ones(cv_img_5.shape[0])
Y_5 = Y_5*(5)
print(Y_5)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S6/*.gif")
cv_img_6 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_6.append(first_gray)
    
cv_img_6 = np.array(cv_img_6)
print(cv_img_6.shape)
cv_img_6 = cv_img_6.reshape(cv_img_6.shape[0], -1)
print(cv_img_6.shape)

Y_6 = np.ones(cv_img_6.shape[0])
Y_6 = Y_6*(6)
print(Y_6)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S7/*.gif")
cv_img_7 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_7.append(first_gray)
    
cv_img_7 = np.array(cv_img_7)
print(cv_img_7.shape)
cv_img_7 = cv_img_7.reshape(cv_img_7.shape[0], -1)
print(cv_img_7.shape)

Y_7 = np.ones(cv_img_7.shape[0])
Y_7 = Y_7*(7)
print(Y_7)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S8/*.gif")
cv_img_8 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_8.append(first_gray)
    
cv_img_8 = np.array(cv_img_8)
print(cv_img_8.shape)
cv_img_8 = cv_img_8.reshape(cv_img_8.shape[0], -1)
print(cv_img_8.shape)

Y_8 = np.ones(cv_img_8.shape[0])
Y_8 = Y_8*(8)
print(Y_8)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S9/*.gif")
cv_img_9 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_9.append(first_gray)
    
cv_img_9 = np.array(cv_img_9)
print(cv_img_9.shape)
cv_img_9 = cv_img_9.reshape(cv_img_9.shape[0], -1)
print(cv_img_9.shape)

Y_9 = np.ones(cv_img_9.shape[0])
Y_9 = Y_9*(9)
print(Y_9)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S10/*.gif")
cv_img_10 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_10.append(first_gray)
    
cv_img_10 = np.array(cv_img_10)
print(cv_img_10.shape)
cv_img_10 = cv_img_10.reshape(cv_img_10.shape[0], -1)
print(cv_img_10.shape)

Y_10 = np.ones(cv_img_10.shape[0])
Y_10 = Y_10*(10)
print(Y_10)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S1/*.gif")
cv_img_11 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_11.append(first_gray)
    
cv_img_11 = np.array(cv_img_11)
print(cv_img_11.shape)
cv_img_11 = cv_img_11.reshape(cv_img_11.shape[0], -1)
print(cv_img_11.shape)

Y_11 = np.ones(cv_img_11.shape[0])
Y_11 = Y_11*(11)
print(Y_11)

path = glob.glob("Desktop/IITD Sem-2/ML/Ass_2/yalefaces/S2/*.gif")
cv_img_12 = []

for img in path:
    
    cap = cv2.VideoCapture(img)
    _,first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.resize(first_gray, (120, 100))
    cv_img_12.append(first_gray)
    
cv_img_12 = np.array(cv_img_12)
print(cv_img_12.shape)
cv_img_12 = cv_img_12.reshape(cv_img_12.shape[0], -1)
print(cv_img_12.shape)

Y_12 = np.ones(cv_img_12.shape[0])
Y_12 = Y_12*(12)
print(Y_12)


# In[ ]:





# In[ ]:




