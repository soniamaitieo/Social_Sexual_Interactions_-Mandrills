#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:11:53 2021
@author: sonia
"""

import os
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import tensorflow_addons as tfa
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer

#--- DATA PROCESSING ---#


#Convert path to (img, label) tuple
def get_label(file_path, class_names):
  parts = tf.strings.split(file_path, os.path.sep)  # convert the path to a list of path components
  one_hot = parts[-2] == class_names  # The second to last is the class-directory
  return tf.argmax(tf.cast(one_hot, tf.int32))

# To process the image
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  resized_image = tf.image.resize(img, [224, 224])
  final_image = tf.keras.applications.vgg16.preprocess_input(resized_image)
  return final_image

def process_path(file_path,class_names):
  label = get_label(file_path,class_names)
  img = tf.io.read_file(file_path)  # load the raw data from the file as a string
  img = decode_img(img)
  return img, label


def process_filename_label(filename, label):
    #filename = tf.read_file(filename)
    img = tf.io.read_file(filename)  # load the raw data from the file as a string
    img = decode_img(img)
    return img, label


def choose_ds(list_ds , choice, image_count,n_train,n_val,n_test):
    if choice =="close":
        #CLOSE-DS#
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        train_ds = list_ds.take(n_train)
        val_ds = list_ds.skip(n_train).take(n_val)
        test_ds = list_ds.skip(n_train + n_val).take(n_test)
        return(train_ds,val_ds,test_ds)
    if choice =="semiclose":
        #SEMI-CLOSE-DS#
        test_ds = list_ds.take(n_test)
        close_ds = list_ds.skip(n_test).take(n_train + n_val)
        #close_ds = close_ds.shuffle(n_train + n_val , reshuffle_each_iteration=False)
        close_ds = close_ds.shuffle(n_train + n_val , reshuffle_each_iteration=True)
        train_ds = close_ds.take(n_train)
        val_ds = close_ds.skip(n_train).take(n_val)
        return(train_ds,val_ds,test_ds)
    if choice =="open":
        #OPEN-DS#
        train_ds = list_ds.take(n_train)
        val_ds = list_ds.skip(n_train).take(n_val)
        return(train_ds,val_ds,test_ds)


def dict_set_ind_img(train_ds,val_ds,test_ds):
    """
    Return 1 dict of 3 dict (train, val,test) with number of images for each ds
    """
    label_tmp = [y.numpy() for x, y in train_ds]
    train_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) )
    label_tmp = [y.numpy() for x, y in val_ds]
    val_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) )
    label_tmp = [y.numpy() for x, y in test_ds]
    test_dict = dict((x, label_tmp.count(x)) for x in set(label_tmp) )
    d = {
        "train": train_dict,
        "val": val_dict,
        "test" : test_dict,
    }
    return(d)

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def dict_ind_img(train_ds,val_ds,test_ds):
    """
    Return 1 dict of 3 dict (train, val,test) with list of all images for each indiv for each ds
    """
    label_tmp = [y.numpy() for x, y in train_ds]
    train_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) )
    label_tmp = [y.numpy() for x, y in val_ds]
    val_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) )
    label_tmp = [y.numpy() for x, y in test_ds]
    test_dict = dict((x, duplicates(label_tmp, x)) for x in set(label_tmp) )
    d = {
        "train": train_dict,
        "val": val_dict,
        "test" : test_dict,
    }
    return(d)


def zoom(x: tf.Tensor) -> tf.Tensor:
    #https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/#zooming
    """Zoom augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(224, 224))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def augment(image, label):
    rd_angle = round(rd.uniform(0, 0.5), 2)
    img = tfa.image.transform_ops.rotate(image, rd_angle)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, 0.3)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    #NEW add translation
    HSHIFT, VSHIFT = 10., 10. # max. number of pixels to shift(translation) horizontally and vertically
    img= tfa.image.translate(img , [HSHIFT * tf.random.uniform(shape=[],minval=-1, maxval=1), VSHIFT * tf.random.uniform(shape=[],minval=-1, maxval=1)]) # [dx dy] shift/translation
    img = zoom(img)
    return (img, label)


AUTOTUNE = tf.data.experimental.AUTOTUNE


# optimze preprocessing performance
def configure_for_performance(ds,n_train,batch_size):
  ds = ds.cache()
  ds = ds.map(augment, num_parallel_calls=AUTOTUNE) # augmentation call
  ds = ds.batch(batch_size=batch_size,drop_remainder=True)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def configure_for_performance_testds(ds):
  ds = ds.cache()
  #mettre drop_remainder=True https://github.com/tensorflow/tensorflow/issues/46123
  ds = ds.batch(64)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


################################################################################
##### make new file VGGmodel.py
from keras_vggface.vggface import VGGFace

def model_vgg16_verif(dr):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vgg16_verif_2tasks(dr,class_names_1,class_names_2):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    new_model = tf.keras.models.Model(inputs=vgg_model.get_input_at(0), outputs=vgg_model.layers[-3].get_output_at(0))
    x = new_model.output
    x=tf.keras.layers.Dropout(dr,name='do')(x)
    x = tf.keras.layers.Dense(128, activation=None,name='class1')(x)
    lambdaa = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    predictions_2 = tf.keras.layers.Dense( len(class_names_2),  activation='softmax', name = "class2")(lambdaa)
    modelcomb = tf.keras.models.Model(inputs=new_model.input, outputs=[x,predictions_2])
    modelcomb.summary()
    for layer in modelcomb.layers[:-4]:
        layer.trainable = False
    for layer in modelcomb.layers:
        print(layer, layer.trainable)
    return(modelcomb)


def model_vgg16_classif(dr,class_names):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vgg16_classif_Mand(dr,class_names):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    model = tf.keras.models.Sequential(vgg_model.layers[:-2])
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    #for layer in model.layers[:-8]:
    for layer in model.layers[:-7]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def model_vgg16_classif_Mand2(dr, dr2,class_names):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    fc7 = vgg_model.layers[-4]
    fc7_relu = vgg_model.layers[-3]
    model = tf.keras.models.Sequential(vgg_model.layers[:-4])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(fc7)
    model.add(fc7_relu)
    model.add(tf.keras.layers.Dropout(dr2))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-7]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vgg16_classif_Mand2_batchnorm(dr, dr2, class_names):
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    fc7 = vgg_model.layers[-4]
    fc7_relu = vgg_model.layers[-3]
    model = tf.keras.models.Sequential(vgg_model.layers[:-4])
    model.add(tf.keras.layers.Dropout(dr))
    model.add(fc7)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(fc7_relu)
    model.add(tf.keras.layers.Dropout(dr2))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-9]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vggmandrill_verif(dr,prev_model):
    #prev_model =  load_pretrained_model(MandrillFaceDir)
    fc7 = prev_model.layers[-6]
    fc7_relu = prev_model.layers[-4]
    #new_model = tf.keras.models.Model(inputs=input, outputs=x)
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-8].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(fc7)
    model.add(fc7_relu)
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def model_vggmandrill_verif2(dr,prev_model):
    #prev_model =  load_pretrained_model(MandrillFaceDir)
    #new_model = tf.keras.models.Model(inputs=input, outputs=x)
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(tf.keras.layers.Dropout(dr))
    x = new_model.output
    x=tf.keras.layers.Dropout(dr,name='do')(x)
    x = tf.keras.layers.Dense(128, activation=None,name='class1')(x)
    lambdaa = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    # Predictions for each task
    # Predictions for each task
    #predictions_1 = tf.keras.layers.Dense(len(class_names_1), activation='softmax', name = "class1")(x)
    predictions_2 = tf.keras.layers.Dense( len(class_names_2),  activation='softmax', name = "class2")(lambdaa)
    modelcomb = tf.keras.models.Model(inputs=new_model.input, outputs=[x,predictions_2])
    modelcomb.summary()
    for layer in modelcomb.layers[:-10]:
        layer.trainable = False
    for layer in modelcomb.layers:
        print(layer, layer.trainable)
    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vggmandrill_classif(dr,class_names,prev_model):
    #prev_model =  load_pretrained_model(MandrillFaceDir)
    fc7 = prev_model.layers[-6]
    fc7_relu = prev_model.layers[-4]
    #new_model = tf.keras.models.Model(inputs=input, outputs=x)
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-8].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(fc7)
    model.add(fc7_relu)
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def model_vggmandrill_classif2(dr,class_names,prev_model):
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def model_vggmandrill_classif3(dr,class_names,prev_model):
    #setseed here for DO : https://stackoverflow.com/questions/49175704/tensorflow-reproducing-results-when-using-dropout
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(tf.keras.layers.Dropout(dr , seed=0))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature' , kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1)))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Dense(len(class_names),  activation='softmax'))
    model.summary()
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model_vggmandrill_classif_2tasks(dr,class_names_1,class_names_2, prev_model):
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    x = new_model.output
    x=tf.keras.layers.Dropout(dr,name='do')(x)
    x = tf.keras.layers.Dense(128, activation=None,name='feature')(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    # Predictions for each task
    predictions_1 = tf.keras.layers.Dense(len(class_names_1), activation='softmax', name = "class1")(x)
    predictions_2 = tf.keras.layers.Dense( len(class_names_2),  activation='softmax', name = "class2")(x)
    modelcomb = tf.keras.models.Model(inputs=new_model.input, outputs=[predictions_1,predictions_2])
    modelcomb.summary()
    for layer in modelcomb.layers[:-10]:
        layer.trainable = False
    for layer in modelcomb.layers:
        print(layer, layer.trainable)
    return(modelcomb)

def model_vggmandrill_verif_2tasks(dr,class_names_1,class_names_2, prev_model):
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    x = new_model.output
    x=tf.keras.layers.Dropout(dr,name='do')(x)
    x = tf.keras.layers.Dense(128, activation=None,name='class1')(x)
    lambdaa = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    # Predictions for each task
    # Predictions for each task
    #predictions_1 = tf.keras.layers.Dense(len(class_names_1), activation='softmax', name = "class1")(x)
    predictions_2 = tf.keras.layers.Dense( len(class_names_2),  activation='softmax', name = "class2")(lambdaa)
    modelcomb = tf.keras.models.Model(inputs=new_model.input, outputs=[x,predictions_2])
    modelcomb.summary()
    for layer in modelcomb.layers[:-10]:
        layer.trainable = False
    for layer in modelcomb.layers:
        print(layer, layer.trainable)
    return(modelcomb)


def model_vggmandrill_classif_centerloss(dr,class_names,prev_model):
    new_model = tf.keras.models.Model(inputs=prev_model.get_input_at(0), outputs=prev_model.layers[-4].get_output_at(0))
    new_model.summary()
    model = tf.keras.models.Sequential()
    model.add( new_model)
    model.add(tf.keras.layers.Dropout(dr , seed=0))
    model.add(tf.keras.layers.Dense(128, activation=None,name='feature' , kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1)))
    model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.add(tf.keras.layers.Embedding(len(class_names),128))
    model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(tf.keras.backend.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss'))
    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def load_pretrained_model(model_dir):
    model = tf.keras.models.load_model(model_dir + "/my_model")
    print("Architecture custom")
    print(model.summary())
    # -8 si defreeze le dernier bloc de conv
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


#def model for making model MandrillFace
#Step 1 train as usual but train/val set
#en utilisant classiquement model_vgg16_classif -> on freeze les 4 dernières couches

#Step 2 train en récupérant le modele précédent

def model0_for_mandrills_defreezeFC(model,dr):
    # Ca vient de : model = load_pretrained_model(model_dir)
    #freeze FC
    print("hello")
    for layer in model.layers[:-8]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)


def model0_for_mandrills_defreezeConv(model,dr):
    #freeze FC
    for layer in model.layers[:-18]:
        layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

def model0_for_mandrills_defreezeAll(model,dr):
    #freeze FC
    #for layer in model.layer:
        #layer.trainable = True
    for layer in model.layers:
        print(layer, layer.trainable)
    return(model)

################################################################################
#### topNacc.py & mapNacc.py

import sklearn.metrics.pairwise
from keras.callbacks import Callback
from keras import Model


def calc_TOPacc(distance_array, val_ds, TOP ):
    y = np.concatenate([y for x, y in val_ds], axis=0)
    #print(y)
    TOPacc = []
    #labels = valid_generator.filenames
    for i in range(len(distance_array)):
        L=[]
        #i = 1
        distance_i = distance_array[i,:]
        #idx = np.argpartition(distance_i, TOP )
        L=  list(np.argpartition(distance_i,range(TOP+1))[1:TOP+1])
        #This returns the k-smallest values. Note that these may not be in sorted order.
        #if i in L: L.remove(i)
        #print("current_indiv:" , y[i])
        #print("Top_3_img" , L)
        #L = list(distance_i[idx[:TOP]])
        #print(itemgetter(*L)(labels))
        #print("Top_3_indiv" , [y[j] for j in L])
        if y[i] in [y[j] for j in L]:
            TOPacc.append(1)
        else:
            TOPacc.append(0)
    return(sum(TOPacc)/len(TOPacc))


all_TOP3_acc = []
## test compared to train images
class TopX(Callback):
    def __init__(self, x):
          self.x = x
          #self.topX = []
    def on_train_begin(self, logs={}):
          self.topX = []
    def on_epoch_end(self, epoch, logs={}):
        TOP = 5
        #distance_array = compute_distance(self.model,self.x)
        distance_array = sklearn.metrics.pairwise_distances(self.model.predict(self.x,verbose=1), metric='l2')
        # = sklearn.metrics.pairwise.cosine_similarity(self.x,self.x)
        #print(self.x.filenames)
        topX = calc_TOPacc(distance_array, self.x, TOP = TOP)
        #print("Top {TOP} - Accuracy(%):   ", round(topX*100,1))
        print(' Top {} - Accuracy {} %'.format(TOP, round(topX*100,1)))
        all_TOP3_acc.append(round(topX*100,1))
        #print(topX)




# We make a dictionnary where the index is the number of the individual, and the value is a list with the position of associated images into a list
def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


#This function sample the image-anchor for the query indiv and 1 image for each indiv
def sample_candidates_dist(curr_ind , y_dict):
    pict_for_each_ind = dict((k, rd.choice(v)) for k, v in y_dict.items())
    Q = pict_for_each_ind[curr_ind] #GET IMG POS OF CURR_IND (this is the "posititive image")
    all_A = y_dict[curr_ind][:] #to avoid mutation in original dict add [:]
    all_A.remove(Q) #take another diff image Q != A
    A = rd.choice(all_A)
    A = (curr_ind,A)
    #return dict of candidates (key=ind,value=pos_img) and the tupe anchor (ind,pos_img)
    return((pict_for_each_ind,A))


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


def calc_ap_per_ind_stackcontrol(res_dir, res_val,curr_ind, y_dict, meta_ds, indiv_test, imgs, TOP=5):
    #get candidates immg
    all_img, anchor = sample_candidates_dist(curr_ind, y_dict = y_dict)
    stack_ancre = meta_ds.Id_full[meta_ds.abs_path == [i for i in imgs if (i.split('/')[-2] in indiv_test)][anchor[1]]]
    stack_pos = meta_ds.Id_full[meta_ds.abs_path == [i for i in imgs if (i.split('/')[-2] in indiv_test)][all_img[curr_ind]]]
    #print("STACKS FOR " + str(curr_ind) + ":" + ("SAME" if stack_pos.values[0] == stack_ancre.values[0] else "DIFF"))
    outF = open(res_dir +"/myStackTestFile.txt", "a")
    outF.write("STACKS FOR " + str(curr_ind) + ":" + ("SAME" if stack_pos.values[0] == stack_ancre.values[0] else "DIFF"))
    outF.write("\n")
    outF.close()
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


#mAP_history=[]
class map_N(Callback):
    def __init__(self, x, TOP):
        self.x = x
        self.TOP = TOP
        self.mAP_history = []
    def on_train_begin(self, logs={}):
        self.all100_mAP = []
        #self.TOP = TOP
        self.mAP_history = []
    def on_epoch_end(self, epoch, logs={}):
        res_val = self.model.predict(self.x,verbose=1) #embedding
        y_label = list(np.concatenate([y for x, y in self.x], axis=0))
        y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
        y_label
        for it in range(100) :
            mAP_each_ind= [calc_ap_per_ind(res_val,i, y_dict,TOP=self.TOP) for i in list(y_dict.keys())]
            self.all100_mAP.append(np.mean(mAP_each_ind))
        print(' MAP@{} - Accuracy {} %'.format(self.TOP, round(np.mean(self.all100_mAP)*100,1)))
        #mAP_history.append(round(np.mean(self.all100_mAP)*100,1))
        self.mAP_history.append(round(np.mean(self.all100_mAP)*100,1))


class map_N_classif(Callback):
    def __init__(self, x, TOP):
        self.x = x
        self.TOP = TOP
        self.mAP_history = []
    def on_train_begin(self, logs={}):
        self.all100_mAP = []
        self.mAP_history = []
    def on_epoch_end(self, epoch, logs={}):
        feature_network = Model(self.model.input, self.model.get_layer('feature').output)
        res_val = feature_network.predict(self.x)
        #res_val = self.model.predict(self.x,verbose=1) #embedding
        y_label = list(np.concatenate([y for x, y in self.x], axis=0))
        y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
        y_label
        for it in range(100) :
            mAP_each_ind= [calc_ap_per_ind(res_val,i, y_dict,TOP=self.TOP) for i in list(y_dict.keys())]
            self.all100_mAP.append(np.mean(mAP_each_ind))
        print(' MAP@{} - Accuracy {} %'.format(self.TOP, round(np.mean(self.all100_mAP)*100,1)))
        self.mAP_history.append(round(np.mean(self.all100_mAP)*100,1))



##############################################################################

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.001
  if epoch > 20:
    learning_rate = 0.0001
  if epoch > 70:
    learning_rate = 0.00001
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)



##############################################################################

def plot_loss_MAPacc(hist , save_path, mAP_history_train, mAP_history_val):
    save_path =  save_path + "/"
    #LOSS
    train_loss=hist.history['loss']
    val_loss = hist.history['val_loss']
    xc=range(len(train_loss))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Train_loss vs Val_loss')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
    #ACC
    plt.figure(2,figsize=(7,5))
    plt.plot(xc, mAP_history_train)
    plt.plot(xc, mAP_history_val)
    plt.ylim(ymin=0)
    plt.xlabel('num of Epochs')
    plt.ylabel('mAP@ accuracy')
    plt.title('Train_acc vs Val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')
    plt.close()

def plot_acc_cls_Man(hist, save_path):
    save_path =  save_path + "/"
    train_acc=hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc=range(len(train_acc))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train_accuracy vs Val_accuracy (weighted accuracy)')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'weighted_accuracy.png')
    plt.close()
    #sparse acc
    train_acc=hist.history['sparse_categorical_accuracy']
    val_acc = hist.history['val_sparse_categorical_accuracy']
    xc=range(len(train_acc))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('sparse_categorical_accuracy')
    plt.title('Train_accuracy vs Val_accuracy (non weighted sparse accuracy)')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'accuracy.png')


def plot_acc_loss_Man_2tasks(hist, save_path):
    save_path =  save_path + "/"
    train_acc_indiv=hist.history['class1_sparse_categorical_accuracy']
    val_acc_indiv = hist.history['val_class1_sparse_categorical_accuracy']
    train_acc_sexe=hist.history['class2_sparse_categorical_accuracy']
    val_acc_sexe = hist.history['val_class2_sparse_categorical_accuracy']
    xc=range(len(train_acc_indiv))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc_indiv,label="Train Indiv")
    plt.plot(xc,val_acc_indiv,label="Val Indiv")
    plt.plot(xc,train_acc_sexe,label="Train Sexe")
    plt.plot(xc,val_acc_sexe,label="Val Sexe")
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train_accuracy vs Val_accuracy ')
    plt.grid(True)
    plt.legend(loc=0)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'accuracy.png')
    plt.close()
    #loss acc
    train_acc=hist.history['loss']
    val_acc = hist.history['val_loss']
    train_acc_indiv=hist.history['class1_loss']
    val_acc_indiv = hist.history['val_class1_loss']
    train_acc_sexe=hist.history['class2_loss']
    val_acc_sexe = hist.history['val_class2_loss']
    xc=range(len(train_acc_indiv))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc,label="Train")
    plt.plot(xc,val_acc,label="Val")
    plt.plot(xc,train_acc_indiv,label="Train Indiv")
    plt.plot(xc,val_acc_indiv,label="Val Indiv")
    plt.plot(xc,train_acc_sexe,label="Train Sexe")
    plt.plot(xc,val_acc_sexe,label="Val Sexe")
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train_loss vs Val_loss ')
    plt.grid(True)
    plt.legend(loc=0)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'loss.png')
    plt.close()

def plot_acc_loss_Hum_2tasks(hist, save_path):
    save_path =  save_path + "/"
    train_acc_indiv=hist.history['class1_sparse_categorical_accuracy']
    val_acc_indiv = hist.history['val_class1_sparse_categorical_accuracy']
    train_acc_sexe=hist.history['class2_sparse_categorical_accuracy']
    val_acc_sexe = hist.history['val_class2_sparse_categorical_accuracy']
    xc=range(len(train_acc_indiv))
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_acc_indiv,label="Train Indiv")
    plt.plot(xc,val_acc_indiv,label="Val Indiv")
    plt.plot(xc,train_acc_sexe,label="Train Sexe")
    plt.plot(xc,val_acc_sexe,label="Val Sexe")
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train_accuracy vs Val_accuracy ')
    plt.grid(True)
    plt.legend(loc=0)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'accuracy.png')
    plt.close()
    #loss acc
    #train_acc=hist.history['loss']
    #val_acc = hist.history['val_loss']
    train_acc_indiv=hist.history['class1_loss']
    val_acc_indiv = hist.history['val_class1_loss']
    train_acc_sexe=hist.history['class2_loss']
    val_acc_sexe = hist.history['val_class2_loss']
    xc=range(len(train_acc_indiv))
    plt.figure(1,figsize=(7,5))
    #plt.plot(xc,train_acc,label="Train")
    #plt.plot(xc,val_acc,label="Val")
    plt.plot(xc,train_acc_indiv,label="Train Indiv")
    plt.plot(xc,val_acc_indiv,label="Val Indiv")
    plt.plot(xc,train_acc_sexe,label="Train Sexe")
    plt.plot(xc,val_acc_sexe,label="Val Sexe")
    plt.ylim(ymin=0.0)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Train_loss vs Val_loss ')
    plt.grid(True)
    plt.legend(loc=0)
    plt.style.use(['classic'])
    plt.savefig(save_path + 'loss.png')
    plt.close()
##########################################################################


def get_mAP_acc_testds(res_dir,res_val,y_label,TOP,meta_ds, indiv_test, imgs):
    y_dict = dict((x, duplicates(y_label, x)) for x in set(y_label) if y_label.count(x) > 1) #on vire indiv qui ont 1 seule image avec la condition
    #TOP = 5 pour calc mAP@5
    all100_mAP = []
    for it in range(100) :
        #On applique pour chaque indiv le calcule de l' average precision
        #mAP_each_ind liste avec le AP de tous les indivdus
        #print pour controler
        mAP_each_ind= [calc_ap_per_ind_stackcontrol(res_dir=res_dir, res_val=res_val, curr_ind=i, y_dict=y_dict, meta_ds = meta_ds, indiv_test=indiv_test, imgs=imgs , TOP=TOP)for i in list(y_dict.keys())]
        all100_mAP.append( np.mean(mAP_each_ind))
    np.mean(all100_mAP)
    print(" mAP@N TEST:" + str(np.mean(all100_mAP)))


##########################################################################

def write_paramsfile(res_dir, model_dir, n_epochs, batch_size, dataset, data_dir,LR,task,categ,dr,class_names,image_count,dr2):
    #---Write a memo file with parameters used for CNN
    #OPTIM2 = "keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)"
    params_file_path =  res_dir + "/params.txt"
    f = open(params_file_path,'w')
    f.write("base:" + "VGGface VGG16 TOP false" + "\n")
    f.write("weights:" + "VGGface Trained  with:" + str(model_dir) + "\n")
    f.write("task:" + str(task) + "\n")
    f.write("category:" + str(categ) + "\n")
    f.write("dataset:" + str(dataset) + "\n")
    f.write("nb_individus:" + str(len(class_names))+ "\n")
    f.write("nb_images:"+ str(image_count)+ "\n")
    f.write("dataset_dir:" + str(data_dir) + "\n")
    f.write("epochs:" + str(n_epochs) + "\n")
    f.write("batch_size:" + str(batch_size) + "\n" )
    f.write("optim1:" + 'Adam' + "\n")
    f.write("optim1_LR:" + str(LR) + "\n")
    #f.write("optim2_LR:" + str(LR2) + "\n")
    #f.write("no freezen layer:" + str(NOFREEZE) + "\n")
    #f.write("Img_preprocessing" + "rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)")
    #f.write("Img_preprocessing:" + "rescale=1./255, horizontal_flip=False, rotation_range=20,width_shift_range=0.05, height_shift_range=0.05,brightness_range=[0.8,1.2], shear_range=0.1,zoom_range=[0.85,1.15]")
    #f.write("val:" + str(0.20) + "\n")
    #f.write("early_stop:" + str(10) + "\n")
    #f.write("reducelr0.1:" + str(5) + "\n")
    f.write("dropout:" + str(dr) + "\n")
    f.write("dropout:" + str(dr2) + "\n")
    #f.write("HorizontalFlip:" + "True" + "\n")

    f.close()






def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed.
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)

    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))
