from sklearn import metrics
import pickle
import scipy.io as sio
import numpy as np

fwo=open('logits_result','rb')
class_label_1=np.array(pickle.load(fwo))
class_label_2=np.array(pickle.load(fwo))
class_label_ema=np.array(pickle.load(fwo))
real_labels=pickle.load(fwo)

label1=class_label_1[:,1]
label2=class_label_2[:,1]
labelema=class_label_ema[:,1]

class_1_auc = metrics.roc_auc_score(real_labels,label1)
print('AUC is : %.4f'%(class_1_auc))

class_2_auc = metrics.roc_auc_score(real_labels,label2)
print('AUC is : %.4f'%(class_2_auc))

ema_auc = metrics.roc_auc_score(real_labels,labelema)
print('AUC is : %.4f'%(ema_auc))