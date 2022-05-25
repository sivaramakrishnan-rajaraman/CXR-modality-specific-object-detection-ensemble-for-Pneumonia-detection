import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # if having multiple GPUS in the system

#%%
#clear warnings and session

import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) 
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

#%%
#import other libraries
import struct
import zlib
import time
import itertools
from itertools import cycle
from matplotlib import pyplot
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from numpy import sqrt
from numpy import argmax
import numpy as np
from scipy import interp
from numpy import genfromtxt
import pandas as pd
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16, DenseNet121, ResNet50, MobileNet, EfficientNetB0, InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow_addons as tfa

#%%
# helper functions: improve resolution of the saved figures

def writePNGwithdpi(im, filename, dpi=(72,72)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   # s = buffer.tostring()
   s = buffer.tobytes()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])
      
#%%
#custom functions to generate confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
# evaluation metrics

def matrix_metrix(real_values,pred_values,beta):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN**2 + FN * TN + FP**2 + FP * TN)
    Prevalence = round( (TP+FP) / Population,2)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    NPV        = round( TN / (TN+FN),4 )
    FDR        = round( FP / (TP+FP),4 )
    FOR        = round( FN / (TN+FN),4 ) 
    check_Pos  = Precision + FDR
    check_Neg  = NPV + FOR
    Recall     = round( TP / (TP+FN),4 )
    FPR        = round( FP / (TN+FP),4 )
    FNR        = round( FN / (TP+FN),4 )
    TNR        = round( TN / (TN+FP),4 ) 
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos      = round( Recall/FPR,4 ) 
    LRNeg      = round( FNR / TNR ,4 )
    DOR        = round( LRPos/LRNeg)
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1
    mat_met = pd.DataFrame({'Metric':['TP','TN','FP','FN','Accuracy','Precision','Recall','F1','MCC','Kappa'],
                            'Value':[TP,TN,FP,FN,Accuracy,Precision,Recall,F1,MCC,Kappa]})
    return (mat_met)

#%%
# load data

img_width, img_height = 512,512
train_data_dir = "data/train" #path to your data
val_data_dir = "data/val"
test_data_dir = "data/est"
epochs = 64 
batch_size = 8
num_classes = 2 
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
# image generators for batch processing

train_datagen = ImageDataGenerator(
        rescale=1./255)

val_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical')

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_val_samples = len(val_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(val_generator.class_indices)
print(test_generator.class_indices)

Y_test=test_generator.classes
print(Y_test.shape)

Y_test1=to_categorical(Y_test, 
                       num_classes=num_classes, 
                       dtype='float32')
print(Y_test1.shape)

#%% if using pretrained model

#models: VGG16, VGG19, DenseNet121, ResNet50, MobileNet, EfficientNetB0, InceptionV3
# here we show how to train a VGG16 model. Repeat for other models. 

vgg16_model = VGG16(weights='imagenet', include_top=False, 
                    input_shape=input_shape) # change for different models
vgg16_model.summary()
x = vgg16_model.output
x1 = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x1)
model_base = Model(inputs=vgg16_model.input, 
                   outputs=predictions, 
                   name='EB0_base_model')
model_base.summary()

#%%
# compute class weights

train_classes = train_generator.classes
class_weights = compute_class_weight(class_weight = "balanced",
                                     classes = np.unique(train_classes),
                                     y = train_classes)
class_weights = dict(zip(np.unique(train_classes), class_weights)),
print(class_weights)

#%%
#enumerate and print layer names

for i, layer in enumerate(model_base.layers):
   print(i, layer.name)    

#%%

# compile and train

opt = SGD(lr=0.0001, momentum=0.9)
model_base.compile(loss='categorical_crossentropy', 
                   optimizer=opt, metrics=['accuracy'])

filepath = 'weights/' + model_base.name +\
            '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='min', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=10, 
                              verbose=1, 
                              mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=10,
                              verbose=1,
                              mode='min', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#reset generators and start training
train_generator.reset()
val_generator.reset()
test_generator.reset()

#train the model
model_history = model_base.fit(train_generator, 
                          steps_per_epoch=nb_train_samples // batch_size,
                          epochs=epochs, 
                          validation_data=val_generator,
                          callbacks=callbacks_list, 
                          validation_steps=nb_val_samples // batch_size, 
                          verbose=1)

print('Training time: %s' % (time.time()-t))

#%%
# plot performance

N = 64 #change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
          model_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("pretrain_performance.png")

#%%

# load model for inference

model = load_model('weights/model1.h5') # path to your saved model
model.summary()
model.compile(loss='categorical_crossentropy', 
                   optimizer=opt, metrics=['accuracy'])

#%%
#Generate predictions on the test data

test_generator.reset() 
custom_y_pred = model.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
#we need the scores of only the positive abnormal class
custom_y_pred1 = custom_y_pred[:,1]

#%%
#print all metrics using the default classification threshold

mat_met = matrix_metrix(Y_test1.argmax(axis=-1),
                      custom_y_pred.argmax(axis=-1),
                      beta=0.4)
print (mat_met)

#%%

# print the confusion matrix

target_names = ['No-finding', 'Abnormal'] 
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=5)
x_axis_labels = ['No-finding', 'Abnormal']  
y_axis_labels = ['No-finding', 'Abnormal'] 
plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', #Greens
            annot_kws={'size': 30},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
#plot the ROC curves

fpr, tpr, thresholds = roc_curve(Y_test, 
                                 custom_y_pred[:,1])
auc_score=roc_auc_score(Y_test, custom_y_pred[:,1])
print(auc_score)

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=2, 
         label='No Skill')
plt.plot(fpr, tpr, 
         marker='.',
         markersize=12,
         markerfacecolor='green',
         linewidth=4,
         color='red',
         label='Model')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%
# plot PR curves

precision, recall, thresholds = precision_recall_curve(Y_test, 
                                 custom_y_pred[:,1])
fscore = (2 * precision * recall) / (precision + recall)

#compute average precision
average_precision_base = average_precision_score(Y_test, 
                                 custom_y_pred[:,1])
print("The average precision value is", average_precision_base)

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall, precision))

# plot the PR curve for the model
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', color='green', label='Model')
# axis labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()

#%%