import numpy as np
import pandas as pd

def random_split(df, train_split = 0.8):
    df = df.reset_index(drop = True)

    np.random.seed(47) #Hace que el train y val sean siempre los mismos

    pos= np.random.permutation(len(df)) 

    div = int(train_split * len(df))

    train_ind = pos[:div]
    val_ind = pos[div:]

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]

    return train_df, val_df

def group_split(df, val_schools):


    mask_val = df["escuela"].isin(val_schools)
    df_train = df[~mask_val].reset_index(drop=True)
    df_val   = df[ mask_val].reset_index(drop=True)


    return df_train, df_val


def temp_split(df, n_train_semesters=5, semester_col="semestre"):
    lim = df[semester_col].drop_duplicates().nsmallest(n_train_semesters).max()

    df_train = df[df[semester_col] <= lim].reset_index(drop=True)
    df_val   = df[df[semester_col] >  lim].reset_index(drop=True)

    return df_train, df_val
