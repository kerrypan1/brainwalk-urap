# URAP-BRAINWALK-Joseph

## Video-to-FGA Fusion Pipeline

This repository contains the code for predicting Functional Gait Assessment (FGA) scores using a multimodal approach, combining clinician-defined "Bath" metrics and automated "Zeno" spatiotemporal metrics.

## Process
### Preprocessing
* labels.py extracts the bath scores (the clinicial metrics)
* zenometric.py extracts the zeno scores
### Bath Metrics
* FGAFinding.py: Main Driver. Script used to train the individual LSTMs for each Bath clinical metric.
* combined.py: Contains the model class definition for the head that creates a linear combination of Bath metrics (to predict FGA)
* FGACombined.py: The execution script that trains the linear combination model to predict the final FGA score from Bath metrics.
* multiple.py: Legacy/Toolbox. Originally used for LSTM training; now serves as a utility import for common functions and constants.
### Zeno Metrics
* zenomodel.py: A singular LSTM designed to predict all 13 Zeno spatiotemporal metrics simultaneously from video.
* zenoCombinedModel.py: A neural head that combines predicted Zeno metrics to generate an FGA prediction.
### Multimodal Fusion
* bathAndZenoModel.py: The final fusion layer that integrates both the Bath and Zeno streams for a unified FGA prediction.
### Evaluation and Visualization
* baselineMAE.py: Calculates the baseline Mean Absolute Error across models.
* everythingErrorBar.py: Performs bootstrapping to calculate confidence intervals and evaluate model stability.
* histogram.py: Generates performance distributions

## General Notes
* The naming convention of the keypoints are ID_DATE_TASK_CAM.npy
* Run FGAFinding.py followed by zenomodel.py before attempting the fusion scripts.
* Currently all of the file paths and csv names are redacted for privacy reasons (altogether possible I have gone overboard but better to do too much than too little in regards to patient data) 
