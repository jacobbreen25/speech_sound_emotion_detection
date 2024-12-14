# Installation
To install, first clone the github 
```git clone git@github.com:jacobbreen25/speech_sound_emotion_detection.git```

Next, install the dataset from [here](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and extract it to your workspace in a folder called data (Should be one directory behind the python files)

# Setup/Creating the Model

First, run the following command which will generate a csv file for the audio feature
```python3 <path to python directory>/generate_training_data.py```

Next, run the following command which will create the model and save it to a .pth file named model_v2-longtrain.pth in the models directory (May need to be made before running)
```python3 <path to python directory>/network_model.py```

# Run existing Model

To run using an existing model, you will first need a .pth file names model_v2-longtrain.pth.

Onece you have this, run the following command to run the model with test data
```python3 <path to python directory>/run_model.py```

# Run live demo

Uou will first need a .pth file names model_v2-longtrain.pth.


To run the live demo, simply run the following command
```python3 <path to python directory>/live_audio.py```

# Emotion Detection Based on Sound and Speech
### Andrew Boland | 01876265, Jacob Breen | 01972052, Peter Gavriel | 01417984,
### Mathew Langlois | 01879896, Ryan Politis | 01980638


## Introduction 
As we progress further into Human Computer Interactions (HCI) it becomes increasingly important to treat the user as an individual. This can be seen in many instances, such as ChatGPT which will attempt to answer a user's questions while also providing good customer service. In models that may be distributed in fields such as customer service, it is important for an individual's mental state to be taken into account. For example, a customer service worker who is refusing customer service may receive an angry response and can account for that response and provide different services to mediate the situation. An automated system cannot adapt to user needs in this way without first being capable of evaluating the emotional state of the user. To this end we propose training a model that will predict the emotional state of a person speaking relying solely on using audio data as its input. 
## Problem Definition
In this project we aim to create a near real-time emotional state predictor that makes predictions based on audio data inputs. Our plan is to train several linear regression models to work in parallel (one vs. all approach), each one predicting the presence of one emotion, with the highest prediction across all of the models being the overall prediction. Alternatively, depending on the performance we are able to achieve, we may also attempt to utilize clustering or train a multiclass neural network classifier on our data to see if we can achieve higher accuracy on our test set. This may end up being necessary as the one vs. all approach assumes independence between each class, which is not the case for emotions. 
The first task is to identify and extract which features of the audio data will be most useful for our task. Since using audio data for machine learning tasks is already well explored, a pre-existing python library named pyAudioAnalysis will help us implement extracting various features. The features that this library can extract from the audio are the Zero Crossing Rate (ZCR), entropy, entropy of energy, spectral centroid, spectral spread, spectral entropy, spectral flux, Mel-Frequency Cepstral Coefficients (MFCCs), chroma vector, and chroma deviation. The ZCR is the measurement of how often a signal changes positive to negative, or to 0. This is useful for filtering out the parts with human speech, and removing gaps where they aren’t speaking. Our goal then becomes investigating how effectively each of these features can correctly predict the presence of each emotion. We will take each feature and find a way to visualize the data, to see which of the different features seem to correlate with each different emotion the best. We will then adjust our features and hyperparameters to create as accurate a model as possible, and then test it with novel data. 
## Dataset 
We plan to leverage the audio-only portion of the RAVDESS Dataset [1] which includes 1,440 audio performances by 24 professional actors (12 male and 12 female), each vocalizing two lexically-matched statements emphasizing different emotions. The seven annotated emotions are calm, happy, sad, angry, fearful, surprise, and disgust. The dataset also labels the intensity of the delivery between two discrete levels, and while predicting the intensity of the dominant emotion is out of scope for this project, it may still end up being a useful feature during training. Certain limitations will be imposed by this dataset, while having a variety of actors is a good quality, they all have North American accents, which will impact the models ability to generalize across different accents and languages. Additionally, while only having two phrases be a part of the training data may help the model distinguish the relevant differences between emotions, it may also impact its capacity to accurately predict data where a user is expressing things that are not in the training set. If necessary, we may branch out to try to include data from other datasets that share a similar format.     	
Evaluation Method
For the evaluation of each individual classifier, we will try to minimize the mean squared error (MSE) on our testing data. For the overall performance of the one vs. all approach, we will look at accuracy score, as well as investigate the confusion matrix to determine which emotions are being misclassified most frequently. We will leverage the confusion matrix to help us select more effective features and tune the individual classifiers as needed. Lastly we will test the overall performance in a qualitative manner by implementing a live demo and testing the model ourselves to see how it performs on novel audio data.  
## Timeline
11/5 - 11/12: Determine model/Extract Features

11/13 - 11/26: Model Training

11/27 - 12/4: Analysis

12/5 - 12/12: Buffer week

## References
[1] Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.
