import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names

if __name__ == "__main__":

    data = pd.read_csv("./data/movie100k/data.csv")
    sparse_features = ["user", "item"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=5)
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )

    es = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=100, verbose=2, validation_split=0.2, callbacks=[es])
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test RMSE", round(mean_squared_error(
        test[target].values, pred_ans) ** 0.5, 4))
