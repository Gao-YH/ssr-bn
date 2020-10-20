import os
import glob
import numpy as np
import cv2
import scipy.io as sio

input_data='/media/disk3/'
output_data='data/images/breast-node/'
datasets=['breast-dataset-A']     #data path name of image files
outputsize=128     #image scale size

for dataset in datasets:
    datapath=os.path.join(input_data,dataset)

    def read_data(path,label):
      images=glob.glob(os.path.join(path,'*.*'))

      imagemat=[]
      labels=[]
      for image in images:
          img=cv2.imread(image)

          if len(img.shape)==2:
              temp=np.zeros((img.shape[0],img.shape[1],3))
              temp[:,:,0]=img
              temp[:,:,1]=img
              temp[:,:,2]=img

              img=temp

          img=cv2.resize(img,(outputsize,outputsize),interpolation=cv2.INTER_CUBIC)
          img = img[:, :, (2, 1, 0)]
          imagemat.append(img.tolist())
          labels.append(label)

      return imagemat,labels

    imagemat=[]
    labels=[]
    pos_path=os.path.join(datapath,'1')
    label=1
    tempmat,templabel=read_data(pos_path,label)
    imagemat.extend(tempmat)
    labels.extend(templabel)

    pos_path = os.path.join(datapath, '0')
    label = 0
    tempmat, templabel = read_data(pos_path, label)
    imagemat.extend(tempmat)
    labels.extend(templabel)

    imagemat=np.array(imagemat,dtype=np.uint8)
    labels=np.array(labels,dtype=np.uint8)
    index = np.random.permutation(np.arange(labels.shape[0]))
    imagemat = imagemat[index]
    labels = labels[index]

    imagemat=np.transpose(imagemat,[1,2,3,0])
    labels=np.reshape(labels,[-1,1])

    dict={'X':imagemat,'y':labels}

    sio.savemat(os.path.join(output_data,dataset+'-'+str(outputsize)+'x'+str(outputsize)+'.mat'),dict)
    del imagemat
    del dict

