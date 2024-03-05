from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def gradient_boosting_rhythm_classification(filtered_df):
    # Preparing the data
    X = filtered_df[['age', 'sex', 'heart_rate', 'standard_deviation', 'weight']]
    y = filtered_df['rhythm_classification']
    record_names = filtered_df['record_name'].values  # Capture record names for later use

    # Encoding the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test, train_record_names, test_record_names = train_test_split(
        X, y_encoded, record_names, test_size=0.2, random_state=42
    )

    # Initialize and train the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_classifier.fit(X_train, y_train)

    # Making predictions
    predictions = xgb_classifier.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Inverse transform the predictions back to original labels
    predicted_rhythms = label_encoder.inverse_transform(predictions)
    actual_rhythms = label_encoder.inverse_transform(y_test)

    # Create a DataFrame for displaying predictions alongside actual labels and record names
    predictions_df = pd.DataFrame({
        'Record Name': test_record_names,
        'Actual Rhythm': actual_rhythms,
        'Predicted Rhythm': predicted_rhythms
    })

    # Print evaluation metrics
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Return the DataFrame of predictions for further analysis
    return predictions_df, xgb_classifier, label_encoder
