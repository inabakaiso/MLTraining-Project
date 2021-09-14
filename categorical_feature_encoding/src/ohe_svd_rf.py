import pandas as pd
from sklearn import ensemble, metrics, preprocessing, decomposition
import config
from scipy import sparse

def run(fold):
    df = pd.read_csv(config.FOLDS_PATH)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    #欠損地の補完
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ##One-Hot-Encoder
    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat(
        [df_train[features], df_valid[features]], axis=0
    )
    ohe.fit(full_data[features])
    X_train = ohe.transform(df_train[features])
    X_valid = ohe.transform(df_valid[features])

    ## 主成分分析による次元圧縮
    svd = decomposition.TruncatedSVD(n_components=120)

    # 学習用と検証用のデータセットを結合し学習
    # vstackではaxis=0　方向に結合している
    full_sparse = sparse.vstack((X_train, X_valid))
    svd.fit(full_sparse)

    X_train = svd.transform(X_train)
    X_valid = svd.transform(X_valid)

    model = ensemble.RandomForestClassifier(n_jobs=-1)

    model.fit(X_train, df_train.target.values)

    ## 各データがそれぞれのクラスに所属する確率を返す =>> logits
    ## 出力が1の値を抜き出している => [60000, 2]
    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"{fold}: {auc}")

if __name__ == "__main__":
    for f in range(5):
        run(f)