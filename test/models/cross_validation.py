# Train the Model with Cross-Validation:
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Split data into features and target
X = combined_df[['age', 'sex', 'heart_rate', 'qrs_width']]  # Feature matrix
y = combined_df['rhythm']  # Target variable

# Encode categorical variables if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline that first imputes missing values, then scales the data, and finally fits the model
pipeline = make_pipeline(imputer, scaler, rf_classifier)

# Use the pipeline to train and evaluate the model
# Note: When using a pipeline, you directly fit it on the raw data; the pipeline will handle imputation and scaling automatically
pipeline.fit(X_train, y_train)

# Evaluate on validation set
validation_accuracy = pipeline.score(X_val, y_val)
print(f"Validation Accuracy: {validation_accuracy}")

# Evaluate on test set
test_accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Perform k-fold cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=k_fold, scoring='accuracy')

# Monitor for overfitting
mean_cv_accuracy = scores.mean()
std_cv_accuracy = scores.std()
print(f"Mean Cross-Validation Accuracy: {mean_cv_accuracy}")
print(f"Standard Deviation of Cross-Validation Accuracy: {std_cv_accuracy}")