import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import data as d
import argparse

def load_csv_data(filepath):

    df = pd.read_csv(filepath)
    emotions = df.iloc[:, 0]
    features = df.iloc[: ,1:]
    print(features.shape)
    
    return emotions, features



emotion_colors_labels ={
    0: 'black', 
    1: 'gray', 
    2: 'cyan', 
    3: 'orange', 
    4: 'blue', 
    5: 'red', 
    6: 'purple', 
    7: 'green', 
    8: 'yellow'
}

legend_labels = {
    "silence": 'black', 
    "neutral": 'gray', 
    "calm": 'cyan', 
    "happy": 'orange', 
    "sad": 'blue', 
    "angry": 'red', 
    "fearful": 'purple', 
    "disgust": 'green', 
    "surpised": 'yellow'
}

emotion_colors = [
    'black', 
    'gray', 
    'cyan', 
    'orange', 
    'blue', 
    'red', 
    'purple', 
    'green', 
    'yellow'
]

def plot2D(features, mapped_colors):
    plt.scatter(x=features[:, 0], y=features[:, 1], c=mapped_colors, s=30, alpha=0.7)
    plt.title = "t-SNE visualization MFCC data"
    xaxis_title="x",
    yaxis_title="y",
    plt.savefig("test.png")

def plot3D(features, mapped_colors, feature_name, title, elev=30, azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=features[:, 0], 
        ys=features[:, 1], 
        zs=features[:, 2], 
        c=mapped_colors, 
        s=5, 
        alpha=0.7
    )
    ax.set_title(f"t-SNE Visualization of {feature_name} Data") 
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=f"{emotion}") 
          for emotion, color in legend_labels.items()]
    ax.legend(handles=handles, title="Emotions", loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_name", help="the name of the feature", type=str)
    parser.add_argument("csv_path", help= "the path for the csv", type=str)
    args = parser.parse_args()

    emotions, feature = load_csv_data(args.csv_path)
    #print(feature)

    #print("Emotions shape:", emotions.shape)  # Should match the number of samples
    #print("MFCC Features shape:", feature.shape)

    mapped_colors = [emotion_colors[emotion] for emotion in emotions]
    scalar = StandardScaler()
    normalized_features = scalar.fit_transform(feature)
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    feature_tsne = tsne.fit_transform(normalized_features)
    print(feature_tsne.shape)

    plot3D(feature_tsne, mapped_colors, args.feature_name, f"./plots/{args.feature_name}.png")
    plot3D(feature_tsne, mapped_colors, args.feature_name, f"./plots/{args.feature_name}1.png", elev=60, azim=120)
    plot3D(feature_tsne, mapped_colors, args.feature_name, f"./plots/{args.feature_name}2.png", elev=20, azim=45)

   

        
    


    

    

