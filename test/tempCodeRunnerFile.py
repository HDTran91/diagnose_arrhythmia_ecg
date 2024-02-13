# Predict probabilities for the test set
# y_pred_probs = model.predict(X_test_aug)

# # Convert predicted probabilities to class labels for multi-class classification
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Calculate the confusion matrix
# conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)

# tn, fp, fn, tp = conf_matrix.ravel()

# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)

# # Calculate AUC
# auc_score = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')

# # Print metrics
# print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
# print(f'Sensitivity: {sensitivity:.4f}')
# print(f'Specificity: {specificity:.4f}')
# print(f'AUC: {auc_score:.4f}')