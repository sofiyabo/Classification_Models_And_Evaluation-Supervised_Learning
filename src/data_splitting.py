import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import preprocessing as prep
import models as mdls
import metrics as mtr
import utils as utls


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

def random_kfold(df, target_col, k=5, random_state=42):

    rng = np.random.default_rng(random_state) #ver bien el uso de esto 
    classes = df[target_col].unique()
    folds = [[] for _ in range(k)]

    for cls in classes:
        cls_idx = df[df[target_col] == cls].index.to_numpy()
        cls_idx = rng.permutation(cls_idx)
        splits  = np.array_split(cls_idx, k)
        for i, split in enumerate(splits):
            folds[i].extend(split.tolist())

    result = []
    for i in range(k):
        val_idx   = np.array(folds[i])
        train_idx = np.concatenate([np.array(folds[j]) for j in range(k) if j != i])
        df_train  = df.loc[train_idx].reset_index(drop=True)
        df_val    = df.loc[val_idx].reset_index(drop=True)
        result.append((df_train, df_val))

    return result

def group_kfold(df, k=8):
   
    schools      = sorted(df["escuela"].unique())
    school_groups = np.array_split(schools, k)
    result       = []

    for val_group in school_groups:
        mask_val = df["escuela"].isin(val_group)
        df_train = df[~mask_val].reset_index(drop=True)
        df_val   = df[ mask_val].reset_index(drop=True)
        result.append((df_train, df_val, list(val_group)))

    return result 



def cv_lambda_random(df, feature_cols, target_col, feats_imp, exclude, lambdas, k=5, random_state=42):

    folds = random_kfold(df, target_col=target_col, k=k, random_state=random_state)
    
    f1_lambdas = []

    for lam in lambdas:
        f1_folds = []

        for df_train, df_val in folds:
            meds = prep.medians(df_train, feats_imp)
            df_train = prep.impute(df_train, meds, feats_imp)
            df_val = prep.impute(df_val,   meds, feats_imp)
            params  = prep.params_norm(df_train, exclude)
            df_train = prep.normalize_df(df_train, params)
            df_val = prep.normalize_df(df_val,   params)

            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            X_val = df_val[feature_cols].values
            y_val = df_val[target_col].values

            model = mdls.LogRegressionL2(lam=lam)
            model.set_model(X_train, y_train)
            y_pred  = model.predict(X_val)

            f1_folds.append(mtr.f1(y_val, y_pred))

        f1_lambdas.append(np.mean(f1_folds))

    return np.array(f1_lambdas)


def cv_lambda_group(df, feature_cols, target_col, feats_imp, exclude, lambdas, k=8):

    folds = group_kfold(df, k=k)
    f1_lambdas = []

    for lam in lambdas:
        f1_folds = []

        for df_train, df_val, _ in folds:
            meds     = prep.medians(df_train, feats_imp)
            df_train = prep.impute(df_train, meds, feats_imp)
            df_val   = prep.impute(df_val,   meds, feats_imp)
            params   = prep.params_norm(df_train, exclude)
            df_train = prep.normalize_df(df_train, params)
            df_val   = prep.normalize_df(df_val,   params)

            X_train = df_train[feature_cols].values
            y_train = df_train[target_col].values
            X_val   = df_val[feature_cols].values
            y_val   = df_val[target_col].values

            model   = mdls.LogRegressionL2(lam=lam)
            model.set_model(X_train, y_train)
            y_pred  = model.predict(X_val)

            f1_folds.append(mtr.f1(y_val, y_pred))

        f1_lambdas.append(np.mean(f1_folds))

    return np.array(f1_lambdas)


def plot_lambda_search(lambdas, f1_random, f1_group, f1_temporal):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, f1_random,   label="Random KFold",  marker='o', markersize=3)
    ax.plot(lambdas, f1_group,    label="Group KFold",   marker='o', markersize=3)
    ax.axhline(f1_temporal.max(), linestyle='--', label=f"Temporal (best={f1_temporal.max():.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("F1 (validación)")
    ax.set_title("F1 vs λ por estrategia de splitting")
    ax.legend()
    plt.tight_layout()
    plt.show()

def cv_coeficientes_random(df, feature_cols, target_col, feats_imp, exclude, lam, k=5):
    folds = random_kfold(df, target_col=target_col, k=k)  # df, no indices
    coefs = []

    for df_train, df_val in folds:
        meds     = prep.medians(df_train, feats_imp)
        df_train = prep.impute(df_train, meds, feats_imp)
        df_val   = prep.impute(df_val,   meds, feats_imp)
        params   = prep.params_norm(df_train, exclude)
        df_train = prep.normalize_df(df_train, params)

        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values

        model = mdls.LogRegressionL2(lam=lam)
        model.set_model(X_train, y_train)
        coefs.append(model.weights.copy())  # weights, no coef_

    return np.array(coefs)  # shape (k, n_features)


def cv_coeficientes_group(df, feature_cols, target_col, feats_imp, exclude, lam, k=8):
    folds = group_kfold(df, k=k)  # df, no groups
    coefs = []

    for df_train, df_val, _ in folds:  # unpack los 3 valores
        meds     = prep.medians(df_train, feats_imp)
        df_train = prep.impute(df_train, meds, feats_imp)
        params   = prep.params_norm(df_train, exclude)
        df_train = prep.normalize_df(df_train, params)

        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values

        model = mdls.LogRegressionL2(lam=lam)
        model.set_model(X_train, y_train)
        coefs.append(model.weights.copy())

    return np.array(coefs)  # shape (k, n_features)