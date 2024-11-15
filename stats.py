import os
import re
import sys
from enum import Enum
from scipy.io import wavfile
import argparse

class Emotion(Enum):
    NEUTRAL = 1
    CALM = 2
    HAPPY = 3
    SAD = 4
    ANGRY = 5
    FEARFUL = 6
    DISGUST = 7
    SUPRISED = 8

class Data:
    def __init__(self, emotion, intensity, statement, repetition, actor, dataPath):
        #01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
        self.emotion = Emotion(emotion)
        #01 = normal, 02 = strong || neutral emotion only has normal intensity
        self.intensity = intensity
        #01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
        self.statement = statement
        #01 = 1st repetition, 02 = 2nd repetition
        self.repetition = repetition
        #01 to 24. Odd numbered actors are male, even numbered actors are female
        self.actor = actor

        self.fs_wav, self.data_wav = scipy.io.wavefile.read(dataPath)


    def __str__(self):
        return f"actor: {self.actor}, repetition: {self.repetition}, statement: {self.statement}, intensity: {self.intensity}, emotion: {self.emotion}"





def createData(audioFile, filePath):
    print(audioFile)
    reg = "\d\d"
    fields = re.findall(reg, audioFile)
    data = Data(int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]), filePath)
    return data


def createDataList(path):
    actorFolderList = os.listdir(path)

    listOfData = []

    for folder in actorFolderList:
        newPath = path + '/' + folder
        dataForActor = os.listdir(newPath)
        for audioFile in dataForActor:
            listOfData.append(createData(audioFile, newPath))

    return listOfData

#print(createData("03-01-06-01-02-01-12"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filepath", help="filepath to the folder with the data",
                           type=str, required=True)
    args = argparser.parse_args()
    path = args.filepath

    dataList = createDataList(path)

    print(sys.path)
    '''
    for data in dataList:
        print(data)
    '''

    

    

