import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time


filePath = os.path.join(os.getcwd(), 'archive', 'BCCC-CIRA-CIC-DoHBrw-2020.csv')
df = pd.read_csv(filePath, delimiter=',', header=0)

# Limit the dataset to 1000 records
df = df.head(1000)
data = df.drop('Label', axis=1)
target = df['Label']

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
start_time = time.time() # Start time for training
clf.fit(X_train, y_train)
end_time = time.time() # End time for training

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Evaluate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Calculate time taken for testing
test_time = time.time() - end_time

# Print the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print("\nTime taken for testing:", test_time, "seconds")

'''# Print the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)'''


'''sns.countplot(y='Label', data=df, palette='rocket', order=df['Label'].value_counts().index, orient='v')
plt.title('Distribution of Malicious and Benign Labels')
plt.xlabel('Count')
plt.ylabel('Label')
plt.show()'''