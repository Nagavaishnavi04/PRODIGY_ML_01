import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
submission_format = pd.read_csv('sample_submission.csv')

# 2. Data preprocessing (select relevant columns)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Drop missing values in train data
train_data_clean = train_data[features + [target]].dropna()

# 3. Prepare training data
X_train = train_data_clean[features]
y_train = train_data_clean[target]

# 4. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test data
X_test = test_data[features]
test_predictions = model.predict(X_test)

# 6. Prepare the submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Get 'Id' column from test data
    'SalePrice': test_predictions  # Predicted prices
})

# Save the submission file
submission.to_csv('my_submission.csv', index=False)

print("Submission file created successfully!")

# Optional: Evaluate the model (if you want to split the training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Optional: Visualize actual vs predicted on validation set
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_val_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices on Validation Set')
plt.show()
