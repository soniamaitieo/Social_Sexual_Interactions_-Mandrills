#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:59:06 2021

@author: sonia
"""


#--- LIBRRAIRIES ---#
import sys
import os
import pathlib
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import gc
import sklearn

sys.path.insert(1, '/home/sonia/Perception_Typicality_CNN/srctools')
from make_tfdata_set import *
os.environ['PYTHONASHSEED'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import random as rd
np.random.seed(1)
rd.seed(1)
tf.random.set_seed(1)


#--- GLOBAL PARAMETERS ---#
img_height = 224
img_width = 224
batch_size = 128
n_epochs = 200
LR = 0.001 #cas defreezing
dr = 0.3

dataset = "MFD20220303" # BKB_CIRMF_AUTRES, MFD20220303, "BKBQ3Beh_CIRMF_AUTRES"
categ = "INDIVIDUAL" #GENDER , INDIVIDUAL
categ2 = "GENDER"
addCenterLoss = False
task = "CLASSIFICATION" #VERIFICATION , CLASSIFICATION
MandrillFaceDir = "/home/sonia/Perception_Typicality_CNN/MANDRILL/MandrillFaceModel/model_folder/2021-09-20-17-47" #alternative=None
model_dir = None

os.chdir("/home/sonia/Perception_Typicality_CNN/MANDRILL")
os.getcwd()


#--- SAVE PARAMETERS and INFOS ---#
today =  datetime.date.today()
now = datetime.datetime.now()

#---create_folder for results ---#
todaystr = today.isoformat() + '-' + str(now.hour) + '-' + str(now.minute)
res_dir = "results/"   + task + '_' + categ +'/' + todaystr

os.mkdir(res_dir)

#---create log file to save all steps and outputs
log_file_path = res_dir + "/log_file.txt"
sys.stdout  = mylog = open(log_file_path, 'w')


### Here the database is a subset of the MANDRILLUS FACE DATABASE
# SEE : https://zenodo.org/record/7467318#.ZBM32ICZPq9
#FOR online db you ha
if dataset == "MFD20220303":
    data_dir = '/media/sonia/DATA/BASE_DE_DONNEE_PORTRAITS_20220303'
    meta_ds = pd.read_csv("/media/sonia/DATA/ProjetMandrillus_Photos/Documents/Metadata/MFD_20220303.csv")
    meta_ds2 = meta_ds[((meta_ds['Age'] > 4) & (meta_ds['sexe'] == 'f')  ) | ((meta_ds['Age'] > 10) & (meta_ds['sexe'] == 'm'))]
    meta_ds3 = meta_ds2.append(meta_ds[(meta_ds['Parent_folder'] == "MANDRILLS AUTRES")], ignore_index=True)
    meta_ds3 = meta_ds3[(meta_ds3['0FaceView'] == '0FaceView1') &  ((meta_ds3['1FaceQual'] == '1FaceQual1') | (meta_ds3['1FaceQual'] == '1FaceQual2') | (meta_ds3['1FaceQual'] == '1FaceQual3') )]
    indexNames = meta_ds3[ meta_ds3['Id_folder'] == 'videonet1_mal_juv' ].index
    meta_ds3.drop(indexNames , inplace=True)
    imgs = meta_ds3['Photo_Path'].to_list()
    image_count = len(imgs)

#--- DATA PROCESSING ---#
AUTOTUNE = tf.data.experimental.AUTOTUNE

# FOLDER=INDIV and FILES=IMAGES in jpg format

if categ == "INDIVIDUAL" and categ2 == None:
    top = 5
    labels = [i.split('/')[-2] for i in imgs]
    class_names = np.array(list(set(labels)))
    labels = pd.Series(labels, dtype="category").cat.codes
elif categ == "GENDER" and categ2 == None:
    top = 1
    class_names = np.array(["m","f"])
    labels = pd.Series(meta_ds3.sexe.to_list(), dtype="category").cat.codes
elif categ == "INDIVIDUAL" and categ2 == "GENDER":
    top1=5
    top2=1
    labels1 = [i.split('/')[-2] for i in imgs]
    class_names1 = np.array(list(set(labels1)))
    class_names2 = np.array(["m","f"])
    labels1 = pd.Series(labels1, dtype="category").cat.codes
    labels2 = pd.Series(meta_ds3.sexe.to_list(), dtype="category").cat.codes


#--- WRITE PARAMS FILE ---#
#write_paramsfile(res_dir, model_dir, n_epochs, batch_size, dataset, data_dir,LR,task,categ,dr,class_names,image_count,dr2)
if dataset == "MFD20220303":
    filenames = tf.constant(imgs)
    img_count = len(filenames)
    n_train = int(img_count * 0.8)
    n_val = img_count - n_train
    if categ2 == None:
        list_ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    else:
        #list_ds = tf.data.Dataset.from_tensor_slices((filenames, {'class1':labels1, 'class2':labels2}))
        list_ds = tf.data.Dataset.from_tensor_slices((filenames, {'class1':labels1, 'class2':labels2}))
    list_ds = list_ds.map(process_filename_label)
    train_ds_4emb =  configure_for_performance_testds(list_ds)
    list_ds = list_ds.shuffle(n_train + n_val , reshuffle_each_iteration=False)
    #--- SPLIT ---#
    n_train = int(img_count * 0.6)
    n_val = int(img_count * 0.3)
    n_test = image_count - n_train - n_val #0.2
    train_ds,val_ds,test_ds = choose_ds(list_ds , "semiclose", image_count,n_train,n_val,n_test)
    #--- CONFIGURE FOR PERFORMANCE---#
    train_ds = configure_for_performance(train_ds,n_train,batch_size)
    val_ds = val_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)


if task == "CLASSIFICATION":
    #--- Call the model ---#
    prev_model = load_pretrained_model(MandrillFaceDir)
    if categ2 == None:
        model = model_vggmandrill_classif3(dr,class_names,prev_model)
    else:
        model= model_vggmandrill_verif_2tasks(dr,class_names1,class_names2, prev_model)
    #--- Compile ---#
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss={'class1':tfa.losses.TripletSemiHardLoss(), 'class2': 'sparse_categorical_crossentropy'},
        metrics = {'class1': 'sparse_categorical_accuracy', 'class2': 'sparse_categorical_accuracy'}
        #weighted_metrics=['accuracy']
        )
    #--- Fit ---#
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=n_epochs,
        )
elif task == "VERIFICATION":
    #--- mAP ACCURACY---#
    mAP_history_val = map_N(val_ds,TOP=top)
    mAP_history_train = map_N(train_ds,TOP=top)
    #--- Call the model ---#
    prev_model = load_pretrained_model(MandrillFaceDir)
    model = model_vggmandrill_verif2(dr, prev_model)
    #--- Compile ---#
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tfa.losses.TripletSemiHardLoss())
    #--- Fit ---#
    history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs=n_epochs,
            #class_weight = class_weights,
            #callbacks=[mAP_history_train,mAP_history_val]
            )

#--- PLOT ---#
plot_acc_loss_Man_2tasks(history,res_dir)

#--- PREDICTION ---#

#y_label = list(np.concatenate([y for x, y in test_ds], axis=0))
y_label1 = list(np.concatenate([y['class1'] for x, y in test_ds], axis=0))
y_label2 = list(np.concatenate([y['class2'] for x, y in test_ds], axis=0))
#y_label_train = list(np.concatenate([y for x, y in train_ds_4emb], axis=0))
y_label_train_1 = list(np.concatenate([y['class1'] for x, y in train_ds_4emb], axis=0))
y_label_train_2 = list(np.concatenate([y['class2'] for x, y in train_ds_4emb], axis=0))

#indiv_test = list(set([i.split('/')[-2] for i in imgs_test]))
indiv_train = list(set([i.split('/')[-2] for i in imgs]))

if task == "CLASSIFICATION":
    from keras import Model
    #feature_network = Model(model.input, model.get_layer('feature').output)
    feature_network = Model(model.input, model.get_layer('class1').output)
    res_val = feature_network.predict(test_ds,verbose=1) #embedding
    res_train = feature_network.predict(train_ds_4emb,verbose=1)
    predtrain = model.predict(train_ds_4emb,verbose=1)
    predtest = model.predict(test_ds,verbose=1)
    predsexe = predtest[1]
elif task == "VERIFICATION":
    res_val = model.predict(test_ds,verbose=1) #embedding
    #res_train = model.predict(train_ds_emb,verbose=1) #embedding_train
    res_train = model.predict(train_ds_4emb,verbose=1)


df_files_train = pd.DataFrame({'Model': [i.split('/')[-2] for i in imgs],
                          'Image':[i.split('/')[-1] for i in imgs],
                          'pred_label_indiv':np.argmax(predtrain[0],axis=1),
                          'pred_label_sexe':np.argmax(predtrain[1],axis=1) ,
                          'label_indiv':y_label_train_1 ,
                          'label_sexe':y_label_train_2 ,
                          'proba_f':list(predtrain[1][:,0]),
                           'proba_m':list(predtrain[1][:,1]),

                     })
print(sklearn.metrics.confusion_matrix(y_label2, np.argmax(predsexe,axis=1)))




#---map ACCURACY---#
def calc_ap_per_ind(res_val,curr_ind, y_dict,TOP=5):
    #get candidates immg
    all_img, anchor = sample_candidates_dist(curr_ind, y_dict = y_dict)
    #Get embedding of candidates images
    res_anchor = res_val[anchor[1], :].reshape(1, -1) #reshape to have compatible dim to compare
    res_comp = res_val[list(all_img.values()),:]
    candidates_comp = list(all_img.keys())
    distance_array_with_A = (sklearn.metrics.pairwise_distances(res_comp, res_anchor, metric='l2')).ravel()
    #Attention L-top va donner le rang des elements qui ne match pas avec le numéro de l'individu, d'où la necessité de garder les num individus dans candidates_comp
    L_top= np.argpartition(distance_array_with_A ,range(TOP))[:TOP]
    tp_counter = 0
    cumulate_precision = 0
    av_precision = 0
    for i in range(len(L_top)):
        #check si l'autre img de lindiv est dans le TOP
        if anchor[0] == candidates_comp[L_top[i]]:
            #compte le nombre de match
            tp_counter += 1
            cumulate_precision += (float(tp_counter)/float(i+1))
    if tp_counter != 0:
        av_precision = cumulate_precision/1 #here we have only 1 positive img
    else:
        av_precision = 0
    return(av_precision)

#MAP VAL
y_dict = dict((x, duplicates(y_label1, x)) for x in set(y_label1) if y_label1.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
all100_mAP = []
for it in range(1000) :
    mAP_each_ind=[calc_ap_per_ind(res_val,i, y_dict,TOP=5) for i in list(y_dict.keys())]
    all100_mAP.append(np.mean(mAP_each_ind))
np.mean(all100_mAP)
print(" mAP@N VAL:" + str(np.mean(all100_mAP)))

#MAP TRAIN
y_dict_train = dict((x, duplicates(y_label_train_1, x)) for x in set(y_label_train_1) if y_label_train_1.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
all100_mAP = []
for it in range(1000) :
    mAP_each_ind=[calc_ap_per_ind(res_train,i, y_dict_train ,TOP=5) for i in list(y_dict_train.keys())]
    all100_mAP.append( np.mean(mAP_each_ind))
np.mean(all100_mAP)
print(" mAP@N TRAIN:" + str(np.mean(all100_mAP)))

#SAVE
np.savetxt(res_dir + "/" + dataset  + "_labelTRAIN.tsv", df_files_train, delimiter=';',fmt='%s')
np.savetxt(res_dir + "/" + dataset + "_embTRAIN.tsv", res_train, delimiter='\t')
np.savetxt(res_dir + "/" + dataset + "_embTRAIN_2.tsv", predtrain[0], delimiter='\t')
modeltosave = res_dir + "/my_model"
model.save(modeltosave)

print('Time Execution')
print(datetime.datetime.now() - now)
