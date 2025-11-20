from ../app/predict import predict

def test_predict_returns_valid_species():
    features = [5.1, 3.5, 1.4, 0.2]
    result = predict(features)

    valid_species = {"Setosa", "Versicolor", "Virginica"}
    assert result in valid_species, f"Unexpected prediction: {result}"
