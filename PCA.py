#!/usr/bin/env python
# coding: utf-8

# In[18]:


# get all path of different images
def get_path_list(list_of_persons_to_consider,remaining_persons):
    if len(list_of_persons_to_consider) > 5:
        print("sorry our database don't have " + len(list_of_persons_to_consider) + " different faces")
        return
    else:
        path_list = []
        all_paths = []
        str1 = "784_assignment2_data/S"
        str2 = "/*.tif"
        for i in list_of_persons_to_consider:
            path_list.append(str1+str(i)+str2)
            all_paths.append(str1+str(i)+str2)
        
        for i in remaining_persons:
            all_paths.append(str1+str(i)+str2)
        
        remaining_paths = list(set(all_paths) - set(path_list))
        return np.array(path_list),np.array(remaining_paths)


# In[19]:


# read images
def read_images(path_list,number_of_train_photo):
    i = 0
    for path in path_list: 
        i = i + 1
        cv_img = []
        cv_img_test = []
        counter = 0
        for img in glob.glob(path):
            counter = counter + 1
            n = cv.cvtColor(cv.imread(img), cv.COLOR_BGR2GRAY)
            if counter > number_of_train_photo:
                cv_img_test.append(n)
            else:
                cv_img.append(n)
        
        if i == 1:
            temp_images = np.array(cv_img)
            temp_images_test = np.array(cv_img_test)
            if len(path_list) == 1:
                return temp_images,temp_images_test
        elif i == 2:
            images = np.concatenate((temp_images, np.array(cv_img)), axis=0)
            images_test = np.concatenate((temp_images_test, np.array(cv_img_test)), axis=0)
        else:
            images = np.concatenate((images,np.array(cv_img)),axis=0)
            images_test = np.concatenate((images_test, np.array(cv_img_test)), axis=0)
            
    return images,images_test


# In[26]:


from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import time
import glob
from sklearn.cluster import KMeans
import seaborn as sns


pure_test_images = False
number_of_train_photo = 12

# make sure remaining_persons always contain only 1 index if pure_test_images is set to false
list_of_persons_to_consider = [1,5,4]

remaining_persons = list(set([1,2,4,5]) - set(list_of_persons_to_consider))
#this list is for impurity testing and 3 is corrupted data so excluded

path_list, remaining_paths = get_path_list(list_of_persons_to_consider,remaining_persons)

images,test_images = read_images(path_list,number_of_train_photo)
total_images, a, b = np.shape(images)
total_images_test, _, _ = np.shape(test_images)
#print("total images "+ str(total_images))
#print("total images test "+str(total_images_test))


remaining_path_images, remaining_path_test_images = read_images(remaining_paths,number_of_train_photo)

if pure_test_images == False:
    print("adding impurity(i.e testing images which were not in the training) test")
    for i in range(len(remaining_path_test_images)):
        test_images[i] = remaining_path_test_images[i]

X,sum_of_X = images_to_matrix(images)
#print(np.shape(X))

print("We are considering these different faces.Our database has 16 faces of each of these persons")
temp = [0,number_of_train_photo,2*number_of_train_photo,3*number_of_train_photo,4*number_of_train_photo]
fig = plt.figure(figsize=(15, 7))
j = 1
for i in temp:
    if i >= total_images:
        break
    else:
        fig.add_subplot(1,5,j)
        plt.imshow(images[i])
        plt.axis('off')
        j = j + 1
plt.show()


X1 = X
X = X - sum_of_X


U, S, V = np.linalg.svd(X.T,full_matrices=False)
print("U(eigen face/vector matrix) shape",np.shape(U))
print(np.shape(S))
print(np.shape(V))



n = 5
print("Ploting first and last "+ str(n) +" eigen faces")
info = "If we pick axis from starting of PC's(U) these will be significant axis in terms of image recovery(i.e these corresponds to highest eigen values) in laymen terms for our example these starting axis are dominated by pixels which are common to all peoples i.e pixels for nose,eyes,hair and all(as we can see in below eigen faces these parts are focused). Later axis are less important for image recovery and for classification purspose these are axis which are least correlated to each other hence corresponds to some particular class this can also be seen in eigen faces figures last eigen faces are somewhat dominated by single(can be more than 1 for multi class) person face while 1st eigen face has mixed features of peoples."
print(info)
counter = 0
while counter < total_images:
    i = 0
    if counter < n or (counter >= (total_images-n)):
        fig = plt.figure(figsize=(15, 7))
        while i < 5:
            temp = np.reshape(U.T[counter],(a,b))
            fig.add_subplot(1,5,i+1)
            plt.imshow(temp)
            plt.axis('off')
            i = i + 1
    counter = counter + 5
    plt.show()

    
    
    

    
i = 0
test_images_column_vector = []
for i in range(len(test_images)):
    test_images_column_vector.append(test_images[i].flatten() - sum_of_X)
test_images_column_vector = np.array(test_images_column_vector)




#r_list = [1,5,30,150,188]
r_list = [1,5,30,50]




print("original test_images")
fig = plt.figure(figsize=(15, 7))
i = 0
for test in test_images:
    i = i + 1
    fig.add_subplot(1,len(test_images),i)
    plt.imshow(test)
    plt.axis('off')
plt.show()
    


class_of_test_images = []

x = len(list_of_persons_to_consider)
y = len(test_images_column_vector)
z = (int)(y/x)

k = 0
for i in range(x):
    k = k + 1
    for j in range(z):
        if  k == 1 and pure_test_images == False:
            class_of_test_images.append(remaining_persons[0])
        else:
            class_of_test_images.append(list_of_persons_to_consider[i])
class_of_test_images = np.array(class_of_test_images)
print("class_of_test_images",class_of_test_images)

plot_data = []

# recovering images
for i in r_list:
    print("recovering faces from " + str(i) + " significant PC's" )
    fig = plt.figure(figsize=(15, 7))
    j = 0
    correctly_classified = 0
    for test_face in test_images_column_vector:
        j = j + 1
        
        temp = (U[:,:i].T @ test_face) # (26320 * i ).T * (26320*1) = size is i * 1
        #print("size of temp = ",np.shape(temp))
        training_projected = X1 @ (U[:,:i]) # (36 * 26320) * (26320 * i ).T = size is 48 * i

        
        diff = training_projected - temp
        norms = np.linalg.norm(diff, axis=1)
        index = np.argmin(norms)
        
        predicted_class = list_of_persons_to_consider[(int)(index/number_of_train_photo)]
        if predicted_class == class_of_test_images[j-1]:
            correctly_classified = correctly_classified + 1
        else:
            print("WARNING --------wronglly classified------")
            #print("this image")
            #plt.imshow(test_images[j-1])
            #plt.show()
            #print("classified closest to")
            #plt.imshow(images[index])
            #plt.show()
            print("predicted class is ",predicted_class," and original class is ",class_of_test_images[j-1])
        
        face = sum_of_X + U[:,:i] @ (U[:,:i].T @ test_face)
        face = np.reshape(face,(a,b))
        fig.add_subplot(1,len(test_images_column_vector),j)
        plt.imshow(face)
        plt.axis('off')
        accuracy = (correctly_classified/y)*100
    print("accuracy on test data is when we consider ",i," PC's is ",accuracy)
    plt.show()



    
    
#sum_of_X = np.reshape(sum_of_X,(a,b))
#plt.imshow(sum_of_X)


print("some intresting results with different PCA components")
plot_graph(U,0,1,len(list_of_persons_to_consider),test_images_column_vector,2)


# In[20]:


def plot_graph(PCA_components,axis_1,axis_2,total_clusters,test_images_column_vector,axis_3=-1):
    if axis_3 == -1:
        print("Below plot is on two axis.")
        print("Axis 1: ",axis_1," Axis 2: ", axis_2)
        axis_1_data = np.reshape(np.array(PCA_components[:,axis_1]),(len(PCA_components[:,axis_1]),1))
        axis_2_data = np.reshape(np.array(PCA_components[:,axis_2]),(len(PCA_components[:,axis_2]),1))
        data = np.hstack((axis_1_data,axis_2_data))
    else:
        ax = plt.axes(projection ='3d')
        print("Below plot is on three axis.")
        print("Axis 1: ",axis_1," Axis 2: ", axis_2," Axis 3: ",axis_3)
        axis_1_data = np.reshape(np.array(PCA_components[:,axis_1]),(len(PCA_components[:,axis_1]),1))
        axis_2_data = np.reshape(np.array(PCA_components[:,axis_2]),(len(PCA_components[:,axis_2]),1))
        axis_3_data = np.reshape(np.array(PCA_components[:,axis_3]),(len(PCA_components[:,axis_3]),1))
        temp_data = np.hstack((axis_1_data,axis_2_data))
        data = np.hstack((temp_data,axis_3_data))
    
    
    color_list = ['blue','green','red','black','brown']
    color = []
    projected_data_to_scatter = []
    counter = 0
    images_per_cluster = len(test_images_column_vector)/total_clusters
    
    print("axis data shape ",np.shape(data))
    for test_face in test_images_column_vector:
        projected_face = (data.T @ test_face)
        projected_data_to_scatter.append(projected_face)
        
        if counter < images_per_cluster:
            color = color_list[0]
        elif counter < 2*images_per_cluster:
            color = color_list[1]
        elif counter < 3*images_per_cluster:
            color = color_list[2]
        elif counter < 4*images_per_cluster:
            color = color_list[3]
        else:
            color = color_list[4]
        counter = counter + 1
        
        if axis_3 == -1:
            plt.scatter(projected_face[0],projected_face[1],c=color)
        else:
            ax.scatter(projected_face[0],projected_face[1],projected_face[2],c=color)
    
    projected_data_to_scatter = np.array(projected_data_to_scatter)
    #print("shape of projected data",np.shape(projected_data_to_scatter))
    
    kmeans = KMeans(n_clusters=total_clusters).fit(projected_data_to_scatter)
    print("kmeans.cluster_centers_")
    print(kmeans.cluster_centers_)
    
    for i in range(len(kmeans.cluster_centers_)):
        if axis_3 == -1:
            plt.scatter(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],c="yellow")
        else:
            ax.scatter(kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1],kmeans.cluster_centers_[i][2],c="yellow")
    plt.show()
    


# In[21]:


"""
make a matrix whose each column represent one photo
"""
def images_to_matrix(images):
    List = []
    total_images = np.shape(images)[0]
    for i in range(total_images):
        A = images[i].flatten()
        List.append(A)
        if i == 0:
            sum = A
        else:
            sum = sum + A
    return np.array(List),sum/total_images

