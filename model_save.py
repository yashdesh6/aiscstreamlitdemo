import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    'Age': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'Height': [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175]
}

df = pd.DataFrame(data)

X = df[['Age']]
y = df['Height']
model = LinearRegression().fit(X, y)

with open("age_height_model.pkl", "wb") as f:
    pickle.dump(model, f)
