import os
import re
import sys
from enum import Enum
import utilities as util

class Emotion(Enum):
    neutral = 1
    calm = 2
    happy = 3
    sad = 4
    angry = 5
    fearful = 6
    disgust = 7
    suprised = 8

class Data:
    def __init__(self, emotion, MFCC, chroma, SC, ZC, RMSE, ML):
        #01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
        self.emotion = emotion
        #01 = normal, 02 = strong || neutral emotion only has normal intensity
        self.MFCC = MFCC
        #01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
        self.chroma = chroma
        #01 = 1st repetition, 02 = 2nd repetition
        self.SC = SC
        #01 to 24. Odd numbered actors are male, even numbered actors are female
        self.ZC = ZC

        self.RMSE = RMSE

        self.ML = ML

