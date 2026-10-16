import numpy as np
import pandas as pd
import os 
import gc
import librosa



# file paths 

raw_path = os.path.join("data","raw")
processed = os.path.join("data" , "processed" , "hive_features.csv")

classes = {'QueenBee Present':1, 'QueenBee Absent':0}

features = []
labels = []
print('Processing Begin')
for f ,l in classes.items():
    folder_path = os.path.join(raw_path , f)
    print(f'The file path is {f}')
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path , file_name)
            
            y , sr = librosa.load(file_path , sr=None)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr , n_mfcc= 40)
            mfccs_mean = np.mean(mfcc.T , axis=0)
            
            features.append(mfccs_mean)
            labels.append(l)

features_x = np.array(features , dtype=np.float32)
df = pd.DataFrame(features_x)            
df['labels'] = labels
print(df.head(5))

df.to_csv(processed, index=False)
print(f"Success! Saved processed features to {processed} with shape {df.shape}")