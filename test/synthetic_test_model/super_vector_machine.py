# Support Vector Machines for synthetic test set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Handle missing values and encode the target variable for the synthetic test set
X_synthetic_test = synthetic_test_set[['age', 'sex', 'heart_rate', 'qrs_width']]  # Feature matrix
y_synthetic_test = synthetic_test_set['rhythm']  # Target variable

# Handle missing values in the synthetic test set features
imputer = SimpleImputer(strategy='most_frequent')
X_synthetic_test_imputed = imputer.fit_transform(X_synthetic_test)

# Replace 'nan' values in the target variable with 'Unknown'
y_synthetic_test = y_synthetic_test.replace('nan', 'Unknown')

# Initialize label encoder with handle_unknown='ignore'
label_encoder = LabelEncoder()
label_encoder.fit(y_synthetic_test)

# Encode categorical variables in the synthetic test set target variable
y_synthetic_test_encoded = label_encoder.transform(y_synthetic_test)

# Make predictions on the synthetic test set
y_pred_synthetic_test = pipeline.predict(X_synthetic_test_imputed)

# Evaluate the models for rhythm diagnosis on the synthetic test set
accuracy_synthetic_test = accuracy_score(y_synthetic_test_encoded, y_pred_synthetic_test)
precision_synthetic_test = precision_score(y_synthetic_test_encoded, y_pred_synthetic_test, average='weighted', zero_division=0)
recall_synthetic_test = recall_score(y_synthetic_test_encoded, y_pred_synthetic_test, average='weighted')
f1_synthetic_test = f1_score(y_synthetic_test_encoded, y_pred_synthetic_test, average='weighted')
conf_matrix_synthetic_test = confusion_matrix(y_synthetic_test_encoded, y_pred_synthetic_test)

# Print evaluation metrics for rhythm diagnosis on the synthetic test set
print("\nSynthetic Test Set Rhythm Diagnosis Metrics:")
print(f"Accuracy: {accuracy_synthetic_test:.2f}")
print(f"Precision: {precision_synthetic_test:.2f}")
print(f"Recall: {recall_synthetic_test:.2f}")
print(f"F1 Score: {f1_synthetic_test:.2f}")
print("Confusion Matrix:")
print(conf_matrix_synthetic_test)

