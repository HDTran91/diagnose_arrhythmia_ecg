# Support Vector Machines
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Split data into features and target
X = combined_df[['age', 'sex', 'heart_rate', 'qrs_width']]  # Feature matrix
y = combined_df['rhythm']  # Target variable

# Encode categorical variables if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the pipeline with an imputer and a standard scaler
pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), SVC(kernel='linear', random_state=42))

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rhythm = pipeline.predict(X_test)

# Evaluate the models for rhythm diagnosis
accuracy_rhythm = accuracy_score(y_test, y_pred_rhythm)
precision_rhythm = precision_score(y_test, y_pred_rhythm, average='weighted', zero_division=0)
recall_rhythm = recall_score(y_test, y_pred_rhythm, average='weighted')
f1_rhythm = f1_score(y_test, y_pred_rhythm, average='weighted')
conf_matrix_rhythm = confusion_matrix(y_test, y_pred_rhythm)

# Print evaluation metrics for rhythm diagnosis
print("\nRhythm Diagnosis Metrics:")
print(f"Accuracy: {accuracy_rhythm:.2f}")
print(f"Precision: {precision_rhythm:.2f}")
print(f"Recall: {recall_rhythm:.2f}")
print(f"F1 Score: {f1_rhythm:.2f}")
print("Confusion Matrix:")
print(conf_matrix_rhythm)