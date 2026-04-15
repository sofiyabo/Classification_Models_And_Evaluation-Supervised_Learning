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

def undersampling(X, y):
    #separo los indices de la clase mayoritaria (1) y la minoritaria (0)
    idx_may = np.where(y == 1)[0]
    idx_min = np.where(y == 0)[0]

    #me quedo con la misma cantidad de clase 1 que de clase 0, eligiendolos de manera random
    idx_may_sub = np.random.choice(idx_may, size=len(idx_min), replace=False)

    indexes = np.concatenate([idx_may_sub, idx_min]) #uno todos los indices
    np.random.shuffle(indexes)  # mezclar los indices para no tener todas las clases juntas

    return X[indexes], y[indexes]

def oversampling_dup(X, y):
    idx_may = np.where(y == 1)[0]
    idx_min = np.where(y == 0)[0]

    # ir duplicando datos de la clase minoritaria de manera random hasta llegar a la misma cantidad que la mayoritaria
    n = len(idx_may) - len(idx_min)
    idx_min_dup = np.random.choice(idx_min, size=n, replace=True)

    indexes = np.concatenate([idx_may, idx_min, idx_min_dup])
    np.random.shuffle(indexes)

    return X[indexes], y[indexes]

def smote(X, y, k_neighbors=5):
    idx_may = np.where(y == 1)[0]
    idx_min = np.where(y == 0)[0]
    X_min = X[idx_min]

    n= len(idx_may) - len(idx_min)
    
    synthetic = []

    for _ in range(n):
        # muestra aleatoria de la clase minoritaria
        idx = np.random.randint(0, len(X_min))
        muestra = X_min[idx]

        # k vecinos mas cercanos, de la misma clase
        dists = np.linalg.norm(X_min - muestra, axis=1)
        dists[idx] = np.inf  # excluir la muestra misma
        vecinos_idx = np.argsort(dists)[:k_neighbors]

        # elegir un vecino aleatorio
        vecino = X_min[np.random.choice(vecinos_idx)]

        #interpolar
        alpha = np.random.uniform(0, 1)
        new = muestra + alpha * (vecino - muestra)
        synthetic.append(new)

    X_synthetic = np.array(synthetic)
    y_synthetic = np.ones(n, dtype=int) * 0  # pongo ceros porque la clase minoritaria es desaprobado (0)

    X_b = np.vstack([X[idx_may], X[idx_min], X_synthetic])
    y_b = np.concatenate([y[idx_may], y[idx_min], y_synthetic])

    idx_shuffle = np.random.permutation(len(y_b))
    return X_b[idx_shuffle], y_b[idx_shuffle]