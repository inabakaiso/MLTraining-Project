import pandas as pd
from sklearn import linear_model, metrics, preprocessing, ensemble
import config


def run(fold):
    df = pd.read_csv(config.FOLDS_PATH)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # 欠損地の補完
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Label-Encoder
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    X_train = df_train[features].values
    X_valid = df_valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs=1)
    model.fit(X_train, df_train.target.values)

    ## 各データがそれぞれのクラスに所属する確率を返す =>> logits
    ## 出力が1の値を抜き出している => [60000, 2]
    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"{fold}: {auc}")


if __name__ == "__main__":
    for f in range(5):
        run(f)