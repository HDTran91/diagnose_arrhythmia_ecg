# using cross-validation for synthetic test
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Prepare features for prediction
X_synthetic_test = synthetic_test_set[['age', 'sex', 'heart_rate', 'qrs_width']]
y_synthetic_test = synthetic_test_set['rhythm']

# Perform k-fold cross-validation on the synthetic test set
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_synthetic_test, y_synthetic_test, cv=k_fold, scoring='accuracy')

# Monitor for overfitting
mean_cv_accuracy = scores.mean()
std_cv_accuracy = scores.std()
print(f"Mean Cross-Validation Accuracy on Synthetic Test Set: {mean_cv_accuracy}")
print(f"Standard Deviation of Cross-Validation Accuracy on Synthetic Test Set: {std_cv_accuracy}")

# Make predictions using the trained model
predictions_synthetic_test = pipeline.predict(X_synthetic_test)

# Convert numerical predictions to strings if the true labels are strings
if isinstance(synthetic_test_set['rhythm'].iloc[0], str):
    predictions_synthetic_test = label_encoder.inverse_transform(predictions_synthetic_test)

# Evaluate model performance on the synthetic test set
accuracy = accuracy_score(y_synthetic_test, predictions_synthetic_test)
precision = precision_score(y_synthetic_test, predictions_synthetic_test, average='weighted', zero_division=0)
recall = recall_score(y_synthetic_test, predictions_synthetic_test, average='weighted')
f1 = f1_score(y_synthetic_test, predictions_synthetic_test, average='weighted')

print("\nModel Performance on Synthetic Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")