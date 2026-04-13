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
    df["escuela"] = df["escuela"].str.upper()

    return df


def medians(df_train, feats):
    medians = {}
    for col in feats:
        medians[col] = df_train[col].median()
    
    return medians

def impute(df, meds, feats):
    df = df.copy()
    for col in feats:
        df[col] = df[col].fillna(meds[col])
    return df

def params_norm(df_train, exclude = None): # uso exclude para features numericas que no quiero normalizar, como las de ordinal encoding
    if exclude is None:
        exclude = []
    
    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    binary = [c for c in numeric_cols if df_train[c].dropna().isin([0, 1]).all()]
    feats_norm = [c for c in numeric_cols if c not in binary and c not in exclude]
    
    params = {}
    for col in feats_norm:
        mean = df_train[col].mean()
        std  = df_train[col].std()
        params[col] = {"mean": mean, "std": std}
    
    return params

def normalize_df(df, params):
    df = df.copy()
    for col, stats in params.items():
        if stats["std"] != 0:
            df[col] = (df[col] - stats["mean"]) / stats["std"]
        else:
            df[col] = 0.0
    return df

