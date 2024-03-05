#  Random Forest Regressor for rhythm Estimation
from sklearn.ensemble import RandomForestClassifier

# Perform one-hot encoding for the 'rhythm' column
label_encoder = LabelEncoder()
combined_df['rhythm_encoded'] = label_encoder.fit_transform(combined_df['rhythm'])

# Separate the DataFrame into numeric and categorical parts
numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = ['rhythm']  # Add other categorical columns if you have any

# Apply mean imputation to numeric data
numeric_imputer = SimpleImputer(strategy='mean')
combined_df[numeric_columns] = numeric_imputer.fit_transform(combined_df[numeric_columns])

# Apply most frequent imputation to categorical data (for 'rhythm' and any others)
categorical_imputer = SimpleImputer(strategy='most_frequent')
combined_df[categorical_columns] = categorical_imputer.fit_transform(combined_df[categorical_columns])

# Encoding the 'rhythm' column again in case imputation was needed
combined_df['rhythm_encoded'] = label_encoder.fit_transform(combined_df['rhythm'])

# Now, prepare your features (X) and target (y)
X = combined_df[['age', 'sex', 'heart_rate', 'qrs_width']]  # assuming these are your numeric features
y = combined_df['rhythm_encoded']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
