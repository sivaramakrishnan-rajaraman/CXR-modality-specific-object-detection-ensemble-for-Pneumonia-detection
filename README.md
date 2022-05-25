# CXR-modality-specific-object-detection-ensemble-for-Pneumonia-detection
Training and constructing ensembles of RetinaNet-based object detection models initialized with random, ImageNet and CXR modality-specific pretrained weights

# Proposal

Computer-aided detection methods using conventional deep learning (DL) models for identifying pneumonia-consistent manifestations in CXRs have demonstrated superiority over traditional machine learning approaches. However, their performance is still inadequate to aid clinical decision-making. This study improves upon the state of the art as follows. Specifically, we train a DL classifier on large collections of CXR images to develop a modality-specific model. Next, we use this model as the classifier backbone in the RetinaNet object detection network. We also initialize this backbone using random weights and ImageNet-pretrained weights. Finally, we construct an ensemble of the best-performing models resulting in improved detection of pneumonia-consistent findings. Experimental results demonstrate that an ensemble of the top-3 performing RetinaNet models outperformed individual models toward this task, which is markedly higher than the state of the art. 

# Requirements:



# Codes

### Modality-specific pretraining.py :

The code shows how to train and evaluate Deep Learning models on a collection of Chest X-rays to learn CXR modlaity-specific features and improve model convergence. 


### object_detection.py:


