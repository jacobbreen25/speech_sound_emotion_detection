import numpy as np
import pandas
import data

df = pandas.read_csv("data.csv")


def unFlattenFeatures(feature, shape):
    print(f"Raw feature string: {feature}")
    print(f"Parsed feature array: {np.fromstring(feature, sep=' ')}")
    shape = shape.strip("() ")
    shape = tuple(map(int, shape.split(',')))
    feature = feature.strip("[] ") 
    feature = np.fromstring(feature, sep=' ') 
    return feature.reshape(shape)
neutralData = []
calmData = []
happyData = []
sadData = []
angryData = []
fearfulData = []
disgustData = []
suprisedData = []



for _, row in df.iterrows():
    #print(row['MFCC'])
    #print(row['MFCC shape'])
    #mfcc = unFlattenFeatures(row['MFCC'], row['MFCC shape'])
    chroma = unFlattenFeatures(row['chroma'], row['chroma shape'])
    sc = unFlattenFeatures(row['SC'], row['SC shape'])
    zc = unFlattenFeatures(row['ZC'], row['ZC shape'])
    rmse = unFlattenFeatures(row['RMSE'], row['RMSE shape'])
    ml = unFlattenFeatures(row['ML'], row['ML  shape'])
    match row['Emotion']:
        case "neutral":
            data = Data(row['Emotion'], mfcc, chroma, sc, zc, rmse, ml)
            neutralData.append(data)
    
'''    
        case "calm":
        case "happy":
        case "sad":
        case "angry":
        case "fearful":
        case "disgust":
        case "suprised":
    
'''


    

    

