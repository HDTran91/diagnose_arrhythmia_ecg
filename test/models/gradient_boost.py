from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def gradient_boosting_rhythm_classification(filtered_df):

    

    # Preparing the data
    X = filtered_df[['heart_rate', 'p_wave_amplitude', 'p_wave_duration','qrs_duration', 'pr_interval','qt_c','f_wave_frequency',
                    'atrial_rate', 'irregular_rhythm','p_wave_presence','rr_interval_variability','increased_heart_rate','p_wave_normal_morphology']]
    
    # Convert categorical columns to one-hot encoded format
    X = pd.get_dummies(X, columns=['irregular_rhythm', 'p_wave_presence', 'increased_heart_rate', 'p_wave_normal_morphology'])

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

    mismatches = [(i, actual, predicted) for i, (actual, predicted) in enumerate(zip(actual_rhythms, predicted_rhythms)) if actual != predicted]

    for index, actual, predicted in mismatches:
        print(f'Index: {index}, Actual: {actual}, Predicted: {predicted}')

    # Print evaluation metrics
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Return the DataFrame of predictions for further analysis
    return predictions_df, xgb_classifier, label_encoder
