import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_synthetic_test_set(synthetic_test_set, classifier, label_encoder):
    # Assuming synthetic_test_set is a DataFrame containing the generated synthetic test data
    # Fill missing values with a placeholder or appropriate strategy
    synthetic_test_set.fillna("Unknown", inplace=True)

    # Prepare features for prediction
    X_synthetic_test = synthetic_test_set[['age', 'sex', 'heart_rate', 'standard_deviation', 'weight']]

    # Make predictions using the trained model
    predictions_synthetic_test = classifier.predict(X_synthetic_test)

    # Convert numerical predictions to strings if the true labels are strings
    if isinstance(synthetic_test_set['rhythm'].iloc[0], str):
        predictions_synthetic_test = label_encoder.inverse_transform(predictions_synthetic_test)

    # Evaluate model performance
    accuracy = accuracy_score(synthetic_test_set['rhythm'], predictions_synthetic_test)
    precision = precision_score(synthetic_test_set['rhythm'], predictions_synthetic_test, average='weighted', zero_division=0)
    recall = recall_score(synthetic_test_set['rhythm'], predictions_synthetic_test, average='weighted')
    f1 = f1_score(synthetic_test_set['rhythm'], predictions_synthetic_test, average='weighted')

    # Display evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Return comparison DataFrame
    comparison_df = pd.DataFrame({
        'Actual Rhythm': synthetic_test_set['rhythm'],
        'Predicted Rhythm': predictions_synthetic_test
    })

    return comparison_df
