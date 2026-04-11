import numpy as np
import pandas as pd

def train_val_split(df, train_split = 0.8):
    df = df.reset_index(drop = True)

    np.random.seed(47) #Hace que el train y val sean siempre los mismos

    pos= np.random.permutation(len(df)) 

    div = int(train_split * len(df))

    train_ind = pos[:div]
    val_ind = pos[div:]

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]

    return train_df, val_df
