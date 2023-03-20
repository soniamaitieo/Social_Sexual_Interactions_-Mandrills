#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:05:56 2021

@author: sonia
"""


#--- LIBRARIES ---#
import tensorflow as tf

import tensorflow_addons as tfa

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import sys
import matplotlib.pyplot as plt
from keras.callbacks import Callback

import os
import pathlib
import numpy as np
import random as rd
import pandas as pd
import sklearn.metrics.pairwise
from scipy.stats import multivariate_normal
import scipy

sys.path.insert(1, '/home/sonia/Perception_Typicality_CNN/srctools')
from make_tfdata_set import *
sys.path.insert(1, '/home/sonia/Perception_Typicality_CNN/HUMAN')
from calc_typicality import *


sys.path.insert(1, '/home/sonia/Perception_Typicality_CNN/srctools/plot_model/')
from plot_model import plot_model

#--- GLOBAL VAR ---#
img_height, img_width = 224,224
dataset = "CFD_US" #["SCUT","CFD","RAYMOND1", "RAYMOND3","FACES", "CFD_W" , "SCUT_W","SCUT_A", "CASIAmimitrain]
save_emb = True
calc_typicality = True

#--- DATA DIR ---#
if dataset == "CFD_US":
    #data_dir = "/home/sonia/cfd/CFDVersion2.5/Images/CFD"
    data_dir = "/media/sonia/RENOULT secure/CFD/Images/CFD/"
elif dataset == "CFD":
    #data_dir = "/home/sonia/cfd/CFDVersion2.5/Images/CFD"
    data_dir = "/media/sonia/RENOULT secure/CFD/Images"
elif dataset == "SCUT" :
   data_dir = "/media/sonia/RENOULT secure/SCUT-FBP/images"
   #data_dir = "/media/sonia/RENOULT secure/SCUT-FBP/images"
elif dataset == "SCUT_W" :
   data_dir = "/media/sonia/RENOULT secure/SCUT-FBP/images"
elif dataset == "SCUT_A":
   data_dir = "/media/sonia/RENOULT secure/SCUT-FBP/images"
   #data_dir = "/media/sonia/DATA/SCUT-FBP/images"
elif dataset == "RAYMOND1":
    data_dir = "/media/sonia/RENOULT secure/Photos Visages Michel Raymond Dataset 1"
elif dataset == "RAYMOND3":
    data_dir = "/media/sonia/RENOULT secure/Photos Visages Michel Raymond Dataset 3"
elif dataset == "FACES":
    data_dir = "/media/sonia/RENOULT secure/FACES"
elif dataset == "CFD_W":
    data_dir = "/media/sonia/RENOULT secure/CFD/Images/CFD"
elif dataset == "CASIAminitrain":
    data_dir = "/media/sonia/RENOULT secure/CASIA_minitrain"
elif dataset=="Facescrub30":
    data_dir = "/media/sonia/RENOULT secure/facescrub/download/mini_face"


#--- RES DIR ---#
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_GENDER/2021-03-17-17-16"
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_GENDER/2021-04-15-11-46"
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_GENDER/2021-04-29-9-45"
res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_GENDER/2021-04-27-18-0"
res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_INDIVIDUAL/2021-04-19-11-10"
res_dir= "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2021-03-12-8-47"

res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_GENDER/2021-04-15-11-46"

#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_GENDER/2021-05-03-9-54"
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_GENDER/2021-05-05-18-28"
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/CLASSIFICATION_INDIVIDUAL/2021-05-18-10-21"
#res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2021-05-20-10-4"
res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-21-9-12"
res_dir = "/home/sonia/Perception_Typicality_CNN/HUMAN/results/VERIFICATION_INDIVIDUAL/2022-04-25-16-50"


#--- GET TASK AND CATEGORY ---#
task = res_dir.split('/')[-2].split('_')[0]
categ = res_dir.split('/')[-2].split('_')[1]
task2 = "CLASSIFICATION"
categ2 = "GENDER"


#--- MODEL ---#
if "categ2" in globals():
    if task == "CLASSIFICATION":
        model = tf.keras.models.load_model(res_dir + "/my_model")
    elif task == "VERIFICATION":
        model = tf.keras.models.load_model(res_dir + "/my_model")
        #model = tf.keras.models.load_model(res_dir + "/my_model", custom_objects = { 'Loss': tfa.losses.TripletSemiHardLoss },compile=False)
else:
    if task == "CLASSIFICATION":
        model = tf.keras.models.load_model(res_dir + "/my_model")
    elif task == "VERIFICATION":
        model = tf.keras.models.load_model(res_dir + "/my_model", custom_objects = { 'Loss': tfa.losses.TripletSemiHardLoss },compile=False)

#model._layers = model._layers[:-1]
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, style=0, color=True, dpi=96)

#--- DATA PROCESSING GENDER ---#


if dataset == "CFD_US":
    #neutral
    #imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if 'N.jpg' in val]
    #non neutral
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
if dataset == "CFD":
    imgsUS = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir + "/CFD")] for val in sublist if 'N.jpg' in val]
    imgsMR = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir + "/CFD-MR")] for val in sublist if 'N.jpg' in val]
    imgsINDIA = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir + "/CFD-INDIA")] for val in sublist if 'N.jpg' in val]
    imgs = imgsUS + imgsMR + imgsINDIA
if dataset == "SCUT":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
if dataset == "RAYMOND1":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
if dataset == "RAYMOND3":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
if dataset == "FACES":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
    imgs = [x for x in imgs if '_n_' in x]
if dataset == "CFD_W":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if 'N.jpg' in val]
    imgs = [i for i in imgs if 'CFD-W' in i]
if dataset == "SCUT_W":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
    imgs = [i for i in imgs if 'C' in i.split('/')[-1]]
if dataset == "SCUT_A":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
    imgs = [i for i in imgs if 'A' in i.split('/')[-1]]
if dataset == "CASIAminitrain":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpg' in val]
if dataset =="Facescrub30":
    imgs = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(data_dir)] for val in sublist if '.jpeg' in val]
    df = pd.read_csv('/media/sonia/RENOULT secure/facescrub/download/GenderSelf_miniface.csv', sep= ","   )


    


if categ == "INDIVIDUAL":
    top = 5
    if dataset == "CFD":
        cat = [i.split('/')[-2] for i in imgsUS] + [i.split('/')[-1][4:10] for i in imgsMR] + [i.split('/')[-1][4:14] for i in imgsINDIA]
        if categ2 == "GENDER":
            cat2 =[i.split('/')[-2][1] for i in imgsUS] + [i.split('/')[-1][4:14][1] for i in imgsMR] + [i.split('/')[-1][5:6] for i in imgsINDIA]
    elif dataset == "CFD_US":
        cat = [i.split('/')[-2] for i in imgs]
    elif  dataset=='CFD_W':
        cat = [i.split('/')[-2] for i in imgs]
    elif dataset == "SCUT" :
        cat = [i.split('/')[-1].split('.')[0] for i in imgs]
        if categ2 == "GENDER":
            cat2 =  [i.split('/')[-1][1] for i in imgs]
        if categ2 == "GENDER":
            cat2 = [i.split('/')[-1][1] for i in imgs]
    elif dataset == "CASIAminitrain":
        cat = [i.split('/')[-2] for i in imgs]
        if categ2 == "GENDER":
            cat2 = ["f"]*len(cat)
    elif dataset =="Facescrub30":
        cat = [i.split('/')[-2].split('.')[0] for i in imgs]
        if categ2 == "GENDER":
            cat2 = df.set_index('Id').loc[cat].reset_index(inplace=False)['Gender'].to_list()
    elif dataset == "SCUT_W" :
        cat = [i.split('/')[-1].split('.')[0] for i in imgs]
    elif dataset == "SCUT_A":
        cat = [i.split('/')[-1].split('.')[0] for i in imgs]
    elif dataset == "RAYMOND1":
        cat = [i.split('/')[-1].split('.')[0] for i in imgs]
    elif dataset == "RAYMOND3":
        cat = [i.split('/')[-1].split('.')[1] for i in imgs]
    elif dataset == "FACES":
        cat = [i.split('/')[-1].split('.')[0][0:3] for i in imgs]
elif categ == "GENDER":
    top = 1
    if dataset == "CFD":
        cat =[i.split('/')[-2][1] for i in imgsUS] + [i.split('/')[-1][4:14][1] for i in imgsMR] + [i.split('/')[-1][5:6] for i in imgsINDIA]
    elif dataset == "CFD_US" :
        cat = [i.split('/')[-2][1] for i in imgs]
    elif  dataset=='CFD_W':
        cat = [i.split('/')[-2][1] for i in imgs]
    elif dataset == "SCUT" :
        cat = [i.split('/')[-1][1] for i in imgs]
    elif dataset == "SCUT_W":
        cat = [i.split('/')[-1][1] for i in imgs]
    elif dataset == "SCUT_A":
        cat = [i.split('/')[-1][1] for i in imgs]
    elif dataset == "RAYMOND1":
        cat = len(imgs) * ['F']
    elif dataset == "RAYMOND3":
        cat = len(imgs) * ['F']
    elif dataset == "FACES":
        cat = [i.split('/')[-1].split('.')[0][6].upper() for i in imgs]
    elif dataset == "Facescrub30":
        cat=df.set_index('Id').loc[[i.split('/')[-2].split('.')[0] for i in imgs]].reset_index(inplace=False)['Gender'].to_list()

if "categ2" in globals():
    list_ds = tf.data.Dataset.from_tensor_slices((imgs, {'class1':cat, 'class2':cat2}))
else:
    list_ds = tf.data.Dataset.from_tensor_slices((imgs, cat))


#--- DATA PROCESSING ---#

# get the count of image files in the test directory
image_count = len([i for i in list_ds])
AUTOTUNE = tf.data.experimental.AUTOTUNE
list_ds = list_ds.map(process_filename_label)
test_ds = list_ds
test_ds = configure_for_performance_testds(test_ds)


#--- PREDICT TEST FACE EMBEDDING  ---#
if "categ2" in globals():
    y_label1 = list(np.concatenate([y['class1'] for x, y in test_ds], axis=0))
    y_label2 = list(np.concatenate([y['class2'] for x, y in test_ds], axis=0))
    y_label2 = [str(i, 'utf-8') for i in y_label2]
    y_label1 = [str(i, 'utf-8') for i in y_label1]

else:
    y = np.concatenate([y for x, y in test_ds], axis=0)
    y_label = list(y)
    y_label = [str(i, 'utf-8') for i in y_label]





if "categ2" in globals():
        feature_network = Model(model.input, model.get_layer('class1').output)
        res_val = feature_network.predict(test_ds,verbose=1) #embedding
        predtest = model.predict(test_ds,verbose=1)
        predsexe = predtest[1]
else:
    if task == "CLASSIFICATION":
        from keras import Model
        feature_network = Model(model.input, model.get_layer('feature').output)
        res_emb = feature_network.predict(test_ds,verbose=1) #embedding
    elif task == "VERIFICATION":
        res_emb = model.predict(test_ds)


if dataset != "CFD":
    df_files = pd.DataFrame({'Model': [i.split('/')[-2] for i in imgs],
                             'Image':[i.split('/')[-1] for i in imgs],
                             'y_label':y_label}
                          )
    """
   df_files = pd.DataFrame({'Model': [i.split('/')[-2] for i in imgs],
                             'Image':[i.split('/')[-1] for i in imgs],
                             'y_pred':np.argmax(predsexe,axis=1),
                             'y_true': pd.Series(cat2, dtype="category").cat.codes}
                          )
   """
else:
    df_files = pd.DataFrame({'Model': [i.split('/')[-1][4:10] if i.split('/')[-2] != "CFD-INDIA" else  i.split('/')[-1][4:14] for i in imgs],
                         'Image':[i.split('/')[-1] for i in imgs],

                         'y_pred':np.argmax(predsexe,axis=1),
                         'y_true': pd.Series(cat2, dtype="category").cat.codes,
                          'sexe_f_prob': predsexe[:,0]})
#get_mAP_acc_testds(res_emb,y_label,TOP=1)
print(sklearn.metrics.confusion_matrix(pd.Series(y_label2, dtype="category").cat.codes, np.argmax(predsexe,axis=1)))

#np.savetxt(res_dir + "/" + dataset  + "_labelTRAIN.tsv", df_files_train, delimiter=';',fmt='%s')
#np.savetxt(res_dir + "/" + dataset + "_embTRAIN.tsv", res_train, delimiter='\t')

if save_emb == True :
    if "categ2" in globals():
        np.savetxt(res_dir + "/" + dataset + "_label.tsv", df_files, delimiter='\t',fmt='%s')
        np.savetxt(res_dir + "/" + dataset + "_emb.tsv", predtest[0], delimiter='\t')

    else:
        np.savetxt(res_dir + "/" + dataset + "_label.tsv", df_files, delimiter='\t',fmt='%s')
        np.savetxt(res_dir + "/" + dataset + "_emb.tsv", res_emb, delimiter='\t')


def cleanCFDMetadata(sheet_name):
    df = pd.read_excel('/media/sonia/RENOULT secure/CFD/CFD 3.0 Norming Data and Codebook.xlsx',
                   sheet_name= sheet_name, engine='openpyxl')
    df = df.drop([0,1,2,3,4,5])
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(6))
    df = df.reindex(df.index.drop(7))
    df = df[['Model' , 'GenderSelf', 'Feminine' , 'Attractive' ]]
    df = df.dropna()
    return(df)



def cleanSCUTMetadata():
    df = pd.read_csv('/media/sonia/RENOULT secure/SCUT-FBP/labels_SCUT_FBP.csv', header= None )
    df = df.rename(columns={0: "Image", 1: "Attractive"})
    df = df.dropna()
    df['GenderSelf'] = df.Image.str[1]
    df['Model'] = [i.split('/')[-1].split('.')[0] for i in df.Image]
    if categ == "INDIVIDUAL" :
        #df_files.Model = df_files.y_label
        df_files.Model = [i.split('/')[-1].split('.')[0] for i in df_files.Image]
    elif categ == "GENDER" :
        df_files.Model = [i.split('/')[-1].split('.')[0] for i in df_files.Image]
    #MERGE
    df_all = df_files.merge(df, how='left',  sort=False)
    return(df_all)



def cleanRaymond1Metadata():
    df = pd.read_csv('/media/sonia/RENOULT secure/Photos Visages Michel Raymond Dataset 1/panel_1.csv', sep= ",")
    df = df[['ID' , 'attractivite' ]]
    df['GenderSelf'] = len(df.ID) * ['F']
    df = df.rename(columns={"ID": "Model"})
    df = df.dropna()
    df.Model=df.Model.astype(str)
    if categ == "INDIVIDUAL" :
        df_files.Model = df_files.Image
    elif categ == "GENDER" :
        df_files.Model = [i.split('/')[-1].split('.')[0] for i in df_files.Image]
    #MERGE
    df_files.Model = [s.lstrip("0") for s in df_files.Model]
    df_all = df_files.merge(df, how='left',  sort=False)
    return(df_all)

def cleanRaymond3Metadata():
    df = pd.read_csv('/media/sonia/RENOULT secure/Photos Visages Michel Raymond Dataset 3/panel 3 data.txt' ,sep='\t')
    df = df[['ID', 'Femininity' , 'Attractiveness' ]]
    df['GenderSelf'] = len(df.ID) * ['F']
    df = df.rename(columns={"ID": "Model"})
    df = df.dropna()
    df.Model=df.Model.astype(str)
    if categ == "INDIVIDUAL" :
        df_files.Model = df_files.y_label
    elif categ == "GENDER" :
        df_files.Model = [i.split('/')[-1].split('.')[1] for i in df_files.Image]
    #MERGE
    df_files.Model = [s.lstrip("0") for s in df_files.Model]
    df_all = df_files.merge(df, how='left',  sort=False)
    return(df_all)


def cleanFACESMetadata():
    df = pd.read_excel('/media/sonia/RENOULT secure/FACES/Appendix_ PerceivedFaceAttractiveness.xlsx', sheet_name='attr_neutral')
    df = df.rename(columns={"Picture": "Image"})
    df = df.rename(columns={"Face Gender": "GenderSelf"})
    df['GenderSelf'] = df.GenderSelf.str[0]
    df_files.Model = [i.split('/')[-1].split('.')[0] for i in df_files.Image]
    df_all = df_files.merge(df, how='left',  sort=False)
    return(df_all)



def import_beauty_data(dataset):
    #--- DATA DIR ---#
    if dataset == "CFD":
        dfUS = cleanCFDMetadata(sheet_name='CFD U.S. Norming Data')
        dfMR = cleanCFDMetadata(sheet_name='CFD-MR U.S. Norming Data')
        dfINDIA = cleanCFDMetadata(sheet_name='CFD-I INDIA Norming Data')
        dfINDIA.Model=dfINDIA.Model.str.replace('IM','IM-')
        dfINDIA.Model=dfINDIA.Model.str.replace('IF','IF-')
        df = pd.concat([dfUS, dfMR, dfINDIA ], keys=['CFD', 'CFD-MR' , 'CFD-INDIA'],names =['folder','index']).reset_index(level=0)
        df = df_files.merge(df, how='left',  sort=False)
    elif dataset == "CFD_US":
        df = cleanCFDMetadata(sheet_name='CFD U.S. Norming Data')
    elif dataset == "CFD_W":
        df = cleanCFDMetadata(sheet_name='CFD U.S. Norming Data')
        df = df_files.merge(df, how='left',  sort=False)
    elif dataset == "SCUT" :
        df = cleanSCUTMetadata()
    elif dataset == "SCUT_W":
        df = cleanSCUTMetadata()
    elif dataset == "SCUT_A":
        df = cleanSCUTMetadata()
    elif dataset == "RAYMOND1":
        df = cleanRaymond1Metadata()
    elif dataset == "RAYMOND3":
        df = cleanRaymond3Metadata()
    elif dataset == "FACES":
        df = cleanFACESMetadata()
    return(df)



if calc_typicality is True:
    res_emb=res_val
    df_all = import_beauty_data(dataset)
    is_normaldist(res_emb)
    #CALC LL
    df_all["LL"] = calc_LL(res_emb,df_all)
    #df_files["LL"] = calc_LL(res_emb,df_files)
    #df_all = df_files.merge(df, how='left',  sort=False)
    # CALC LL - F

    if (dataset != "RAYMOND1" ) and (dataset != "RAYMOND3"):
        df_F = df_all[df_all['GenderSelf'] == "F"]
        res_emb_F = res_emb[df_F.index]
        is_normaldist(res_emb_F)
        df_F = df_F.reset_index()
        LL_F = calc_LL(res_emb_F,df_F)
        df_F['LL_F'] = LL_F
        df_all = pd.merge(df_all,df_F[['Model','LL_F']], how='left' ,on='Model')
        # CALC LL - M
        df_M = df_all[df_all['GenderSelf'] == "M"]
        res_emb_M = res_emb[df_M.index]
        is_normaldist(res_emb_M)
        df_M = df_M.reset_index()
        LL_M = calc_LL(res_emb_M,df_M)
        df_M['LL_M'] = LL_M
        df_all =  pd.merge(df_all,df_M[['Model','LL_M']], how='left' ,on='Model')
        # DIST CENTROID
        fem_moy = np.mean(res_emb_F, axis=0)
        men_moy = np.mean(res_emb_M, axis=0)
        df_all["dist_centre_F"] = [scipy.spatial.distance.euclidean(res_emb[i,: ],fem_moy) for i in range(len(res_emb))]
        df_all["dist_centre_H"] = [scipy.spatial.distance.euclidean(res_emb[i,: ],men_moy) for i in range(len(res_emb))]
    moy = np.mean(res_emb, axis=0)
    df_all["dist_centre"] = [scipy.spatial.distance.euclidean(res_emb[i,: ],moy) for i in range(len(res_emb))]
    #df_all_2 = df_all.merge(df_files , on = ["Image"])
    df_all.to_csv(res_dir + "/" + dataset + "_typicality_measures.csv")
