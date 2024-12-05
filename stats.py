import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import data as d

df = pandas.read_csv("data.csv")

emotion_colors = {
    "neutral": "blue",
    "calm": "green",
    "happy": "orange",
    "sad": "purple",
    "angry": "red",
    "fearful": "brown",
    "disgust": "pink",
    "suprised": "cyan"
}


def unFlattenFeatures(feature, shape):
    print(f"Raw feature string: {feature}")
    print(f"Parsed feature array: {np.fromstring(feature, sep=' ')}")
    shape = shape.strip("() ")
    shape = tuple(map(int, shape.split(',')))
    feature = feature.strip("[] ") 
    feature = np.fromstring(feature, sep=' ') 
    return feature.reshape(shape)

def prepare_tsne_data(data_points, feature_name):
    features = []
    labels = []
    
    for datapoint in data_points:
        feature = getattr(datapoint, feature_name)
        features.append(feature.flatten())
        labels.append(datapoint.emotion)
    
    return np.array(features), labels

def plot_tsne(features, labels, feature_name, color_map):
    n_samples = len(features)
    perplexity = min(30, n_samples - 1)  


    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    for emotion in set(labels):
        indices = [i for i, label in enumerate(labels) if label == emotion]
        plt.scatter(
            reduced_features[indices, 0], 
            reduced_features[indices, 1], 
            label=emotion, 
            color=color_map[emotion], 
            alpha=0.7
        )
    
    plt.title(f"t-SNE Visualization of {feature_name.capitalize()}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(title="Emotions")
    plt.grid(True)
    plt.savefig("{feature_name.capitalize()}.png")    

if __name__ == "__main__":
    allData = []
    for _, row in df.iterrows():
        #print(row['MFCC'])
        #print(row['MFCC shape'])
        mfcc = unFlattenFeatures(row['MFCC'], row['MFCC shape'])
        chroma = unFlattenFeatures(row['chroma'], row['chroma shape'])
        sc = unFlattenFeatures(row['SC'], row['SC shape'])
        zc = unFlattenFeatures(row['ZC'], row['ZC shape'])
        rmse = unFlattenFeatures(row['RMSE'], row['RMSE shape'])
        ml = unFlattenFeatures(row['ML'], row['ML shape'])
        print(row['Emotion'])
        data = d.Data(row["Emotion"], mfcc, chroma, sc, zc, rmse, ml)
        allData.append(data)

        features = ["MFCC", "chroma", "SC", "ZC", "RMSE", "ML"]

        for feature_name in features:
            feature_data, labels = prepare_tsne_data(allData, feature_name)
            plot_tsne(feature_data, labels, feature_name, emotion_colors)

        
    


    

    

