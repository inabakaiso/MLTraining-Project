
import joblib
import config
import os
import pandas as pd
from sklearn import metrics

import model_dispatcher

import argparse

def run(fold, model):

    # import pdb
    # pdb.set_trace()

    df = pd.read_csv(config.FOLDS_PATH)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #valuesを用いることによってnumpy配列に変換することが可能
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    ##model格納はbin-fileに入れるようにする
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"df_{fold}.bin")
    )

if __name__ == "__main__":
    ## 以下構文を使用することにより指定したfoldを行うことができ、無駄なメモリを消費しない
    ##　コマンドラインから実行する
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model
    )