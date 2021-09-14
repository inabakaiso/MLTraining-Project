import os
import config
import pandas as pd
import joblib
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.models import Model, load_model

def create_model(data, catcols):
    """
    :param data: pandasデータフレーム
    :param catcols: 質的変数の列のリスト
    :return: tf.kerasモデル
    """

    inputs = []
    outputs = []

    for c in catcols:
        #列内のカテゴリ数
        num_unique_values = int(data[c].nunique())

        #　埋め込みの次元数の計算
        #カテゴリ数が十分に大きい場合はある程度の次元数が必要
        #カテゴリ数の半分　or 50の小さい方を採用する
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c
        )(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)
        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation="relu") (x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    y = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)

    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

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

    #モデルの作成
    model = create_model(df, features)

    #データセットの作成
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(
        xtrain,
        ytrain_cat,
        validation_data=(xvalid, yvalid_cat),
        verbose=0,
        batch_size=14,
        epochs=20
    )

    valid_preds = model.predict(xvalid)[:, 1]

    print(metrics.roc_auc_score(yvalid,valid_preds))

if __name__ == "__main__":
    for f in range(5):
        run(f)
