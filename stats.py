import os
import re
from enum import Enum
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
    def __init__(self, emotion, intensity, statement, repetition, actor):
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


    def __str__(self):
        return f"actor: {self.actor}, repetition: {self.repetition}, statement: {self.statement}, intensity: {self.intensity}, emotion: {self.emotion}"





def createData(audioFile):
    reg = "\d\d"
    fields = re.findall(reg, audioFile)
    data = Data(int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]))
    return data

#print(createData("03-01-06-01-02-01-12"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filepath", help="filepath to the folder with the data",
                           type=str, required=True)
    args = argparser.parse_args()
    path = args.filepath()

    actorFolderList = os.listdir(path)

    listOfData = []

    for folder in actorFolderList:
        newPath = path + '/' + folder
        dataForActor = os.listdir(newPath)
        for audioFile in dataForActor:
            listOfData.append(createData(audioFile))

    numNeutral = 0
    numCalm = 0
    numHappy = 0
    numSad = 0
    numAngry = 0
    numFearful = 0
    numDisgust = 0
    numSuprised = 0


    for data in listOfData:
        print(data)
        match data.emotion:
            case Emotion.NEUTRAL:
                numNeutral += 1
            case Emotion.CALM:
                numCalm += 1
            case Emotion.HAPPY:
                numHappy += 1
            case Emotion.SAD:
                numSad += 1
            case Emotion.ANGRY: 
                numAngry += 1
            case Emotion.FEARFUL:
                numFearful += 1
            case Emotion.DISGUST:
                numDisgust += 1
            case Emotion.SUPRISED:
                numSuprised += 1

    print(numNeutral, numCalm, numHappy, numSad, numAngry, numFearful, numDisgust, numSuprised)
    print(numNeutral + numCalm + numHappy + numSad + numAngry + numFearful + numDisgust + numSuprised)
    

