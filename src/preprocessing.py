import numpy as np
import pandas as pd

def ordinal_encoding(df, mapeo, feat): #A diferencia del one hot encoding, de esta manera 
    df = df.copy()
    df[feat] = df[feat].map(mapeo)

    return df

def one_hot_encoding(df, feature): #Uso one hot encoding para la feature escuela
    df = df.copy()
    one_hot  = pd.get_dummies(df[feature], columns=[feature], prefix=feature, drop_first=True, dtype=int) 
    df = pd.concat([df, one_hot], axis =1) 
    return df

def notes_scale(df):
    df = df.copy()
    
    df.loc[df["nota_previa"] >  10, "nota_previa"] = df.loc[df["nota_previa"] > 10, "nota_previa"] / 10
    return df

def school_letters(df):
    df = df.copy()
    df['escuela'] = df['escuela'].str.upper()

    return df

def handle_missing_values(df, feats):
    df = df.copy()

    for col in feats:
        df[col] = df.groupby('escuela')[col].transform(lambda x: x.fillna(x.median()))
    
    return df