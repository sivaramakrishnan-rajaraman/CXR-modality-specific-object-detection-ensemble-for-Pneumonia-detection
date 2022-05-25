# CXR-modality-specific-object-detection-ensemble-for-Pneumonia-detection
Training and constructing ensembles of RetinaNet-based object detection models initialized with random, ImageNet and CXR modality-specific pretrained weights

# Proposal

Computer-aided detection methods using conventional deep learning (DL) models for identifying pneumonia-consistent manifestations in CXRs have demonstrated superiority over traditional machine learning approaches. However, their performance is still inadequate to aid clinical decision-making. This study improves upon the state of the art as follows. Specifically, we train a DL classifier on large collections of CXR images to develop a modality-specific model. Next, we use this model as the classifier backbone in the RetinaNet object detection network. We also initialize this backbone using random weights and ImageNet-pretrained weights. Finally, we construct an ensemble of the best-performing models resulting in improved detection of pneumonia-consistent findings. Experimental results demonstrate that an ensemble of the top-3 performing RetinaNet models outperformed individual models toward this task, which is markedly higher than the state of the art. 

# Requirements:

matplotlib==3.3.4

numpy==1.19.5

opencv_python==4.5.1.48

pandas==1.1.5

scikit_learn==1.1.1

scipy==1.5.4

seaborn==0.11.1

tensorflow==2.4.0

tensorflow_addons==0.12.1

vit_keras==0.1.0


# Codes

### Modality-specific pretraining.py :

The code shows how to train and evaluate DL models on a collection of Chest X-rays to learn CXR modality-specific features and improve model convergence. 


### object_detection.py:


