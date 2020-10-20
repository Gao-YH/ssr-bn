A Semi-supervised breast nodule recognition method
The method was based on the mean teacher model propsed in  [Paper](https://arxiv.org/abs/1703.01780) ---- [NIPS 2017 poster](nips_2017_poster.pdf) ---- [NIPS 2017 spotlight slides](nips_2017_slides.pdf).

run:
prepare_datamat.py:  prepare the dataset of mat file format, all the images were scaled to 128*128.
split_datamat.py: randomly create three data files of train set, extra set(no label) and test set.
train_breast.py: train a semi-supervised model by train set and extra set, and evaluate it.
predict_breast.py:  predict test set  
compute_AUC.py:  calculate AUC from logits_result file

dataset-A: tensorflow/data/images/breast-node 
train.mat: train and validation set using labels
extra.mat: traing set not using labels
test.mat: test set

results/train_breast: pre-trained model

