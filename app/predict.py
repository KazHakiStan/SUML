import joblib
import numpy as np

model = joblib.load('./app/model.joblib')


def predict(features):
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map[prediction[0]]

    return predicted_species
