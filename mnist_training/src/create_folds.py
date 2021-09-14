import config
from sklearn.model_selection import KFold
import pandas as pd

##mnist_train_folds


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_PATH)


    #-1の列を作り初期化する
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = KFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        ## validationになるときの番号をfold行に添付する
        df.loc[val_, 'kfold'] = fold

    df.to_csv('../input/mnist_train_folds.csv', index=False)
