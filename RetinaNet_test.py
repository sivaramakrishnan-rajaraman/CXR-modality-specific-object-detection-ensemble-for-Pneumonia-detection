#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../')
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet import losses
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


# use this to change which GPU to use
gpu = 0
# set the modified tf session as backend in keras
setup_gpu(gpu)

plt.ioff()

"""
# -------------------note ------------------------------------------#
# it assumes the ground truth of the test set is available and plots the ROC curve. 
# it requires an input csv file with image full path and corresponding label, 1: control, 2: case
# It requires inputs of the following paths 
#-------------------------------------------------------------------# 
#-------------------input--------------------------------------#
# test_data             -- the .csv (or other comma delimited file) including all image paths and ground truth labels to be tested.
#                       -- the format should be image full path, ...., label
# model_path            -- the full path of the trained model file that's going to be used
# output_save_folder    -- the folder where the tested images will be saved with the automatic predction marked on images
# rocfig_save_path      -- the file name that the roc curve is going to be saved when all the images are tested
# outputFile            -- the output file save path. The file should record all the tested image namese and the corresponding score
# plot                  -- the boolean variable for plotting ROC
# printFile             -- the boolean variable for saving the output file with predictions
# save_marked_iamge     -- the boolean variable for saving all the tested images with predictions marked
#--------------------------------------------------------------#
"""
"""
#-------------------ouput--------------------------------------#
# the output file will be output with two columns
# column 1: image name
# column 2: the predicted score of that image being "case"
#--------------------------------------------------------------#


"""
## where the .csv list is located containing all the test imagaes
test_data = '/slurm_storage/guop2/Data/Data_RSNA_Siva/RSNA/detection_data/labels_retinanet/test_merge_retinanet.csv' 
## the model path
model_path = '/slurm_storage/guop2/Data/Data_RSNA_Siva/RSNA/slurm_jobs/model_20220130_alldefault_retinanet/snapshots/resnet50_csv_65.h5'

## folder where the marked images are saved, 'test_marked' is created if not existed
output_save_folder = '/your folder'
marked_save_path = os.path.join(output_save_folder, 'test_')
rocfig_save_path = os.path.join(output_save_folder, 'result.png')
# print file which saves the image names and predicted scores for all target categories in each column
outputFile = './output path.csv'


    
# The three boolean variables, that toggle three printing behaviors: 
plot = False                 # plot and save the ROC curve, change the plot to be False/True if plot is unwanted/wanted
printFile = True            # print and save the scores, change 'prinfFile' to be False/True if a .txt file is unwanted/wanted
save_marked_image = False    # save the tested images with predictions marked, change 'save_marked_image' to be False/True if .jpg is unwanted/wanted

"""
#-----------------------------------------------------The method definition starts here:------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------# 
"""


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    


## test each image with the loaded model, and plot, save the marked images if 'save_marked_image' is true
def testOneImage(imgPath = None, save_marked_image = False, saveFolder = None, model = None, threshold = 0.01):
    # load image
    image = read_image_bgr(imgPath)
    imname = imgPath.rstrip().split('/')[-1]
    
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side = 512)
    
    # process image, uncomment if the time caculation is needed.
#    start = time.time()
    boxes, scores, labels, score_all = model.predict_on_batch(np.expand_dims(image, axis=0))
#    print("processing time: ", time.time() - start)
    ind_box = np.where(scores > threshold)
    
    boxes /= scale
    
    t, score, label = boxes[ind_box], scores[ind_box], labels[ind_box]
    
    if save_marked_image:
        for i in range(len(t)): 
            text_pos = t[i]
            text_pos = text_pos.astype(int)
            #box = boxes[0][indScore[1][0], :].astype(int)
            caption = "score {:.3f}".format(score[i])
            draw_caption(draw, text_pos, caption)
            cv2.rectangle(draw, pt1 = (text_pos[0], text_pos[1]), pt2 = (text_pos[2], text_pos[3]), color = [0,200,0], thickness = 2)
        #    
        fig = plt.figure(figsize=(12, 8.18), dpi = 100)
        plt.axis('off')
        plt.imshow(draw)
#       plt.show()
        saveImgPath = os.path.join(saveFolder, imname)
        plt.savefig(saveImgPath)
        plt.close(fig) #so that the figure won't show
    return label, score, score_all, t


def testBatchImage(imgPath = None, save_marked_image = False, saveFolder = None, model = None, batch_size = 2):
    # load image
    img_batch = None
    draw_batch = None
    scale_batch = []
    name_batch = []
    text_pos = []
    score = []
    label = []
    for p in imgPath:
#        imgpath = p.rstrip().split(',')[0]
        image = read_image_bgr(p)
        imname = p.split('/')[-1]
        name_batch.append(imname)
        # copy to draw on
        
        
        
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side = target_h)
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = cv2.resize(image, (target_w, target_h))
#        image, scale = resize_image(image, dsizemin_side = 800)
#        image_resize = cv2.resize(image, (target_w, target_h))
        scale_x = image.shape[1] / draw.shape[1]
        scale_y = image.shape[0] / draw.shape[0]
        scale_batch. append([scale_x, scale_x, scale_y, scale_y])
        if save_marked_image:
            if draw_batch is None:
                draw_batch = np.expand_dims(draw, axis=0)
            else:
                draw_batch = np.concatenate((draw_batch, np.expand_dims(draw, axis=0)), axis = 0)
        if img_batch is None:
            img_batch = np.expand_dims(image, axis=0)
        else:
            img_batch = np.concatenate((img_batch, np.expand_dims(image, axis=0)), axis = 0)
        # process image, uncomment if the time caculation is needed.
#    start = time.time()
    boxes, scores, labels, score_all = model.predict_on_batch(img_batch)
#    print("processing time: ", time.time() - start)
    
#    boxes /= scale
    
    for i in range(img_batch.shape[0]):
        text_pos.append(boxes[i,0,:] / scale_batch[i])
        score.append(scores[i, 0])
        label.append(labels[i, 0])
    
#    text_pos, score, label = boxes[0,0,:], scores[0, :], labels[0,:]
    
    if save_marked_image:
        for ind, text_pos_content in enumerate(text_pos):
            text_pos = text_pos_content.astype(int)
        #box = boxes[0][indScore[1][0], :].astype(int)
            caption = "score {:.3f}".format(score_all[ind][0][1])
            draw_caption(draw_batch[ind], text_pos, caption)
        
        #    
            fig = plt.figure(figsize=(16, 12))
            plt.axis('off')
            plt.imshow(draw_batch[ind])
#       plt.show()
            saveImgPath = os.path.join(saveFolder, name_batch[ind])
            plt.savefig(saveImgPath)
            plt.close(fig) #so that the figure won't show
    return label, score, score_all, name_batch

# plot the confusion matrix according to the predicted labels and ground truth
def conf_matrix(y_true, y_predicted):
    confM = np.zeros([len(np.unique(labelGT)), len(np.unique(labelGT))])
    for i in range(len(y_true)):
        if y_true[i] == y_predicted:
            confM[y_true[i], y_true[i]] += 1
        else:
            confM[y_true[i], y_predicted[i]] += 1
    return confM


# this function pick thresholding samples for plotting the figures
def pick10Thres(tpr, fpr, thres):
    pickX = []
    pickY = []
    pickThres = []
    i = 0
    while i in range(len(tpr) - 1):
        if i == 0:
            i += len(tpr)//10
            continue
        pickX.append(tpr[i])
        pickY.append(fpr[i])
        pickThres.append(thresholds[i])
        i += len(tpr)//10
    return np.asarray(pickX), np.asarray(pickY), np.asarray(pickThres)


"""
#-----------------------------------------------------------------The main code starts here:--------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------# 
"""

conf_mat = None
labelGT=[]
predicted_label = []
predicted_score = []
predicted_score_all = []
batch_size = 1
target_w = 512
target_h = 512

if not os.path.exists(marked_save_path):
    os.mkdir(marked_save_path)
    
import keras
from keras_retinanet import backend
#from tensorflow.keras import backend as K

def focal_tversky_loss(gamma = 1.5, alpha = 0.3):
    
    def _tversky(y_true, y_pred, smooth=1e-10):
        
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)
        
#        y_true_pos = keras.backend.flatten(y_true)
#        y_pred_pos = keras.backend.flatten(y_pred)
        true_pos = keras.backend.sum(labels * classification)
        false_neg = keras.backend.sum(labels * (1-classification))
        false_pos = keras.backend.sum((1-labels)*classification)
#        alpha = alpha # from base paper
        pt_1 = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
        
        return keras.backend.pow((1-pt_1), gamma)
    return _tversky


#from keras.models import load_model

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
  
#model = load_model(model_path, custom_objects = {'focal_tversky_loss': focal_tversky_loss})
  
model = models.convert_model(model)
    
result_recorder = {}



# draw each cervix image with automatic predicted bounding boxes and labels
if batch_size != 1:
    with open(test_data) as csvReader:
        lines = [line.rstrip().split(',')[0] for line in csvReader]
    with open(test_data) as csvReader:
        labelGT = [int(line.rstrip().split(',')[1]) - 1 for line in csvReader]
    startingLine = 0
    total_lines = len(lines)    
    while (startingLine < total_lines): 
        endingLines = startingLine + batch_size - 1
        if endingLines >= total_lines:
            endingLines = total_lines - 1
        desired_lines = lines[startingLine: endingLines + 1]
        desired_gt = labelGT[startingLine: endingLines + 1]
        print(str(startingLine) +  ',' + str(endingLines))
        l, s, s_all, name_batch = testBatchImage(imgPath = desired_lines, save_marked_image = save_marked_image, saveFolder = marked_save_path, model = model, batch_size = batch_size)
        for i in range(len(name_batch)):
            predicted_label.append(l[i])
            predicted_score.append(s[i])
            predicted_score_all.append(s_all[i][0])    
            result_recorder[name_batch[i]] = s_all[i][0]
        startingLine = endingLines + 1
else:
    with open(test_data) as csvReader:
        next(csvReader)
        for r in csvReader:
            imname = r.rstrip().split(',')[0]
#            imname = imgPath.split('/')[-1]
            imgPath = os.path.join('/slurm_storage/guop2/Data/Data_RSNA_Siva/RSNA/detection_data/test_merge', imname)
            gt = 0
            labelGT.append(gt)
            l, s, s_all, box = testOneImage(imgPath = imgPath, save_marked_image = save_marked_image, saveFolder = marked_save_path, model = model, threshold = 0.5)
#            l, s, s_all = testBatchImage(imgPath = imgPath, save_marked_image = save_marked_image, saveFolder = marked_save_path, model = model)
#            predicted_label.append(l[0])
#            predicted_score.append(s[0])
#            predicted_score_all.append(s_all[0][0])    
#            result_recorder[imname] = s
            result_recorder[imname] = box
#conf_mat = conf_matrix(labelGT, predicted_label)   
#score_array = np.asarray(predicted_score_all)

# make the labels and scores into arrays, get them ready for the ROC plot and .txt print
l = np.asarray(labelGT)
#s_copy = score_array[:,1]


### plot the ROC curve
#if plot:
#    import matplotlib.pyplot as plt
#    from sklearn.metrics import roc_curve, auc
#    #    p = np.asarray(s_copy)
#    # the positive label is 1, since here "case" is labeled as "1"
#    fpr, tpr, thresholds  =  roc_curve(l, s_copy, pos_label = 1, drop_intermediate = True)
#    roc_auc =auc(fpr, tpr)
#    plt.figure()
#    lw = 2
#    plt.figure(figsize=(10,10))
#    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([-0.001, 1.01])
#    plt.ylim([-0.001, 1.01])
#    plt.xlabel('False Positive Rate', fontsize = 18)
#    plt.ylabel('True Positive Rate', fontsize = 18)
#    plt.title('ROC', fontsize = 18)
#    plt.legend(loc="lower right", fontsize = 18)
#    ## plot the thresholds
#    pickX, pickY, pickThres = pick10Thres(fpr, tpr, thresholds)
#    pickThres = np.round(pickThres, 2)
#    for i in range(len(pickX)):
#        plt.plot(pickX[i], pickY[i], 'bo', linewidth = 2)
#        plt.text(pickX[i], pickY[i], pickThres[i], fontsize=12)
#
### following line save the figure at 'savePath', is commented since for now no figure is needed to be saved
#    plt.savefig(rocfig_save_path)
### display the figure in the ipython kernel, commented if run the program in a commandline.
##    plt.show() 

if printFile:
    with open(outputFile, 'w') as box_file:
        for im_name in result_recorder.keys():
            bbox = result_recorder[im_name]
#            box_file.write(im_name + "," + str(int(bbox[0])) + "," + str(int(bbox[1])) + "," + str(int(bbox[2])) + "," + str(int(bbox[3])) + "\n" )
            for item in bbox:
                # if format is x, y, h, w
#                box_file.write(im_name + "," + str(int(item[0])) + "," + str(int(item[1])) + "," + str(int(item[2]) - int(item[0])) + "," + str(int(item[3]) - int(item[1])) + "\n" )
                #if format is x1, y1, x2, y2
                box_file.write(im_name + "," + str(int(item[0])) + "," + str(int(item[1])) + "," + str(int(item[2])) + "," + str(int(item[3])) + "\n" )
