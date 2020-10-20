import numpy as np
import scipy.io as sio
import os

input_data='data/images/breast-node/breast-dataset-A-128x128.mat'    #dataset mat file
output_data='data/images/breast-node/'     #ouput random train, val ,extra and test set from dataset

#dataset set size
train_num=[500,500]   #select 20% data for val while train
extra_num=[2559,1826]
test_num=[878,483]

data=sio.loadmat(input_data)
imagemat=data['X']
label=data['y']

imagemat_0=[]
label_0=[]
imagemat_1=[]
label_1=[]

for i in range(len(label)):
    if label[i]==[0]:
        imagemat_0.append(imagemat[:,:,:,i])
        label_0.append([0])

    if label[i]==[1]:
        imagemat_1.append(imagemat[:, :, :,i])
        label_1.append([1])

imagemat_0=np.array(imagemat_0,dtype=np.uint8)
imagemat_1=np.array(imagemat_1,dtype=np.uint8)
label_0=np.array(label_0,dtype=np.uint8)
label_1=np.array(label_1,dtype=np.uint8)

#shuffle 0 and 1 data
num=np.arange(imagemat_0.shape[0])    #get number
index=np.random.permutation(num)
imagemat_0=imagemat_0[index,:,:,:]
label_0=label_0[index]

num=imagemat_1.shape[0]    #get number
index=np.random.permutation(num)
imagemat_1=imagemat_1[index,:,:,:]
label_1=label_1[index]

def save_mat(imagemat,label,mainfile):
    data = {'X': imagemat, 'y': label}
    sio.savemat(os.path.join(output_data,mainfile) + '.mat', data)

def combine_mat(mat0,label0,mat1,label1):
    imagemat=np.concatenate((mat0,mat1),axis=0)
    labels=np.concatenate((label0,label1),axis=0)

    index = np.random.permutation(labels.shape[0])
    imagemat = imagemat[index]
    labels = labels[index]
    imagemat = np.transpose(imagemat, [1, 2, 3, 0])

    return imagemat,labels

#create train data file
train_mat0=imagemat_0[0:train_num[0]]
train_label0=label_0[0:train_num[0]]
train_mat1=imagemat_1[0:train_num[1]]
train_label1=label_1[0:train_num[1]]
train_mat,train_label=combine_mat(train_mat0,train_label0,train_mat1,train_label1)
mainfile='train'
save_mat(train_mat,train_label,mainfile)

#create extra data file
extra_mat0=imagemat_0[train_num[0]:train_num[0]+extra_num[0]]
extra_label0=np.ones(extra_num[0])*2
extra_mat1=imagemat_1[train_num[1]:train_num[1]+extra_num[1]]
extra_label1=np.ones(extra_num[1])*2
extra_mat,extra_label=combine_mat(extra_mat0,extra_label0,extra_mat1,extra_label1)
mainfile='extra'
save_mat(extra_mat,extra_label,mainfile)

#create test data file
test_mat0=imagemat_0[train_num[0]+extra_num[0]:]
test_label0=label_0[train_num[0]+extra_num[0]:]
test_mat1=imagemat_1[train_num[1]+extra_num[1]:]
test_label1=label_1[train_num[1]+extra_num[1]:]
test_mat,test_label=combine_mat(test_mat0,test_label0,test_mat1,test_label1)
mainfile='test'
save_mat(test_mat,test_label,mainfile)
