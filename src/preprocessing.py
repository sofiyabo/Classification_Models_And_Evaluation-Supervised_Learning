import numpy as np
import pandas as pd

def ordinal_encoding(df, mapeo, feat): #A diferencia del one hot encoding, de esta manera 
    df = df.copy()
    df[feat] = df[feat].map(mapeo)

    return df

def one_hot_encoding(df, feature): #Uso one hot encoding para la feature escuela
    df = df.copy()
    one_hot  = pd.get_dummies(df[feature], columns=[feature], prefix=feature, drop_first=True) 
    df = pd.concat([df, one_hot], axis =1) 
    return df
