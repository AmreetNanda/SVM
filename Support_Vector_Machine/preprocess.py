import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_split(df):
    X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
    y = df['total_bill']

    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_time = LabelEncoder()

    X['sex'] = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['time'] = le_time.fit_transform(X['time'])

    ct = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first'), [3])],
        remainder='passthrough'
    )

    X_transformed = ct.fit_transform(X)

    encoders = {
        "le_sex": le_sex,
        "le_smoker": le_smoker,
        "le_time": le_time,
        "ct": ct
    }

    return np.array(X_transformed), np.array(y), encoders


def preprocess_input(encoders, tip, sex, smoker, day, time, size):
    le_sex = encoders["le_sex"]
    le_smoker = encoders["le_smoker"]
    le_time = encoders["le_time"]
    ct = encoders["ct"]

    temp = [[tip,
             le_sex.transform([sex])[0],
             le_smoker.transform([smoker])[0],
             day,
             le_time.transform([time])[0],
             size]]

    return ct.transform(temp)
