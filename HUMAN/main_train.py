#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:21:08 2021

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
import pickle

import pandas as pd

sys.path.insert(1, '/home/sonia/Perception_Typicality_CNN/srctools')
from make_tfdata_set import *


#--- GLOBAL PARAMETERS ---#
img_height = 224
img_width = 224
batch_size = 128
n_epochs = 100
#LR = 0.0001
LR = 0.0001
dr = 0.3

dr2 = None
dataset = "SCRUB" #CASIA20
categ = "INDIVIDUAL" #GENDER INDIVIDUAL
categ2 = "GENDER"
task = "VERIFICATION" #VERIFICATION
model_dir = None
#model_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2021-03-12-8-47"

#--- GET DIR ---#
os.chdir("/home/sonia/Perception_Typicality_CNN/HUMAN")
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


#--- DATA DIR ---#
if dataset == "SCRUB":
    data_dir = "/media/sonia/DATA/facescrub/download/faces"

elif dataset == "CASIA20":
    data_dir = "/media/sonia/DATA/CASIA20"

data_dir = pathlib.Path(data_dir)

#--- DATA PROCESSING ---#
AUTOTUNE = tf.data.experimental.AUTOTUNE

if dataset == "SCRUB":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpeg' in val]
    image_count = len(list(data_dir.glob('*/*.jpeg')))

elif dataset == "CASIA20":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
    image_count = len(list(data_dir.glob('*/*.jpg')))

# FOLDER=INDIV and FILES=IMAGES in jpg format

if categ == "INDIVIDUAL":
    top = 5
    labels = [i.split('/')[-2] for i in imgs]
    class_names = np.array(labels)
    labels = pd.Series(labels, dtype="category").cat.codes
    if ("categ2" in globals() and  dataset == "SCRUB"):
        gender = pd.read_csv('/media/sonia/DATA/facescrub/GenderSelf.csv',index_col=0,header=0)
        class_names_2 = np.array(["F","M"])
        d_gender= dict( zip( gender.actores, pd.Series(gender.GenderSelf, dtype="category")))
        gender.GenderSelf = pd.Series(gender.GenderSelf, dtype="category")
        gender['code'] = gender.GenderSelf.cat.codes
        cat = [i.split('/')[-2] for i in imgs]
        df = pd.DataFrame()
        df['img']  = imgs
        df['actores'] = cat
        df = df.merge(gender)
        labels2 = df.code


elif categ == "GENDER":
    top = 1
    if dataset == "SCRUB":
        gender = pd.read_csv('/media/sonia/DATA/facescrub/GenderSelf.csv',index_col=0,header=0)
        class_names = np.array(["F","M"])
        d_gender= dict( zip( gender.actores, pd.Series(gender.GenderSelf, dtype="category")))
        gender.GenderSelf = pd.Series(gender.GenderSelf, dtype="category")
        gender['code'] = gender.GenderSelf.cat.codes
        imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpeg' in val]
        cat = [i.split('/')[-2] for i in imgs]
        df = pd.DataFrame()
        df['img']  = imgs
        df['actores'] = cat
        df = df.merge(gender)
        labels = df.code



#--- WRITE PARAMS FILE ---#
write_paramsfile(res_dir, model_dir, n_epochs, batch_size, dataset, data_dir,LR,task,categ,dr,class_names,image_count,dr2)

#--- TRANSFORM TO TF_DATA---#

if "categ2" in globals():
    filenames = tf.constant(imgs)
    list_ds = tf.data.Dataset.from_tensor_slices((filenames, {'class1':labels, 'class2':labels2}))
else:
    filenames = tf.constant(imgs)
    labels = tf.constant(labels)
    list_ds = tf.data.Dataset.from_tensor_slices((filenames, labels))


list_ds = list_ds.map(process_filename_label)

#--- SPLIT ---#
n_train = int(image_count * 0.7)
n_val = int(image_count * 0.1)
#n_test= int(n * 0.01)
n_test = image_count - n_train - n_val #0.2
train_ds,val_ds,test_ds = choose_ds(list_ds , "semiclose", image_count,n_train,n_val,n_test)

#--- CONFIGURE FOR PERFORMANCE---#
train_ds = configure_for_performance(train_ds,n_train,batch_size)
val_ds = val_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().batch(batch_size,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)


#--- TRAIN THE MODEL GIVEN THE tASK---#

if task == "CLASSIFICATION":
    #--- mAP ACCURACY---#
    mAP_history_val = map_N_classif(val_ds,TOP=top)
    mAP_history_train = map_N_classif(train_ds,TOP=top)
    #--- Call the model ---#
    if model_dir == None:
        model = model_vgg16_classif(dr,class_names)
    else:
        model = load_pretrained_model(model_dir)
    #--- Compile ---#
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy'])
    #--- Fit ---#
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=n_epochs,
        callbacks = [mAP_history_train,mAP_history_val]
        )
elif task == "VERIFICATION":
    #--- mAP ACCURACY---#
    mAP_history_val = map_N(val_ds,TOP=top)
    mAP_history_train = map_N(train_ds,TOP=top)
    #--- Call the model ---#
    if model_dir == None:
        if "categ2" in globals():
             model =model_vgg16_verif_2tasks(dr,class_names,class_names_2)
        else:
            model = model_vgg16_verif(dr)
    else:
        model = load_pretrained_model(model_dir)
    #--- Compile ---#
    if "categ2" in globals():
        model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss={'class1':tfa.losses.TripletSemiHardLoss(), 'class2': 'sparse_categorical_crossentropy'},
        metrics = {'class1': 'sparse_categorical_accuracy', 'class2': 'sparse_categorical_accuracy'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR),
            loss=tfa.losses.TripletSemiHardLoss())
    #--- Fit ---#
    history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs=n_epochs
            #callbacks=[mAP_history_train,mAP_history_val]
            )


#--- PLOT ---#
#plot_loss_MAPacc(history , res_dir , mAP_history_train.mAP_history, mAP_history_val.mAP_history)
plot_acc_loss_Hum_2tasks(history,res_dir)

#--- GET MAP ACCURACY ON TEST DATASET ---#
if "categ2" in globals():
    y_label1 = list(np.concatenate([y['class1'] for x, y in test_ds], axis=0))
    y_label2 = list(np.concatenate([y['class2'] for x, y in test_ds], axis=0))
else:
    y_label = list(np.concatenate([y for x, y in test_ds], axis=0))


if task == "CLASSIFICATION":
    from keras import Model
    feature_network = Model(model.input, model.get_layer('feature').output)
    res_val = feature_network.predict(test_ds,verbose=1) #embedding
elif task == "VERIFICATION":

    if "categ2" in globals():
        feature_network = Model(model.input, model.get_layer('class1').output)
        res_val = feature_network.predict(test_ds,verbose=1) #embedding
        predtest = model.predict(test_ds,verbose=1)
        predsexe = predtest[1]
    else:
        res_val = model.predict(test_ds,verbose=1) #embedding


#get_mAP_acc_testds(res_val,y_label,TOP=top)

print(sklearn.metrics.confusion_matrix(y_label2, np.argmax(predsexe,axis=1)))


print('Time Execution')
print(datetime.datetime.now() - now)

modeltosave = res_dir + "/my_model"
model.save(modeltosave)

with open(res_dir + '/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


if "categ2" not in globals():
    np.savetxt(res_dir + "/" + dataset + "_label.tsv", y_label, delimiter='\t',fmt='%s')
    np.savetxt(res_dir + "/" + dataset + "_emb.tsv", res_val, delimiter='\t')


#mylog.close()
