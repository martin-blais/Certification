import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_train,y_train)

pickle.dump(rf, open('iris_model.pkl', 'wb'))
