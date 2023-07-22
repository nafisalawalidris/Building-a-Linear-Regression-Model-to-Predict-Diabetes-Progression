#!/usr/bin/env python
# coding: utf-8

# # Import software libraries and load the dataset #

# In[1]:


import sys                                             # Read system parameters.
import numpy as np                                     # Work with multi-dimensional arrays and matrices.
import pandas as pd                                    # Manipulate and analyze data.
import matplotlib as mpl                               # Create 2D charts.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb                                   # Perform data visualization.
import sklearn                                         # Perform data mining and analysis.
from sklearn import datasets

# Summarize software libraries used.
print('Libraries used in this project:')
print('- Python {}'.format(sys.version))
print('- NumPy {}'.format(np.__version__))
print('- pandas {}'.format(pd.__version__))
print('- Matplotlib {}'.format(mpl.__version__))
print('- Seaborn {}'.format(sb.__version__))
print('- scikit-learn {}\n'.format(sklearn.__version__))

# Load the dataset.
diabetes = datasets.load_diabetes()
print('Loaded {} records.'.format(len(diabetes.data)))


# # Get acquainted with the dataset

# In[2]:


import pandas as pd

# Convert array to pandas DataFrame.
df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# View data types and see if there are missing entries.
print(df_diabetes.dtypes)

# View first 10 records.
print(df_diabetes.head(10))


# # Examine the distribution of various features

# In[3]:


# Use Matplotlib to plot distribution histograms for all features.
import matplotlib.pyplot as plt

# Define a color palette for histograms
colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'yellow', 'purple', 'pink', 'lightblue', 'lime', 'salmon', 'lightgray']

# Plot distribution histograms for all features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(df_diabetes.columns):
    plt.subplot(3, 4, i+1)
    plt.hist(df_diabetes[feature], bins=20, color=colors[i], edgecolor='black')
    plt.title(feature)
plt.tight_layout()
plt.show()


# # Examine a general summary of statistics

# In[4]:


# View summary statistics (mean, standard deviation, min, max, etc.) for each feature.
summary_statistics = df_diabetes.describe()
print(summary_statistics)


# # Look for columns that correlate with `target` (disease progression)#

# In[5]:


import numpy as np
import pandas as pd
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Create a DataFrame from the dataset
data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Add the target column 'disease_progression' to the DataFrame
data['disease_progression'] = diabetes.target

# Assign the DataFrame to the variable df_diabetes
df_diabetes = data

# View the first few records of the DataFrame to ensure it's loaded correctly
print(df_diabetes.head())


# In[6]:


# View the correlation values for each feature compared to the label "disease_progression" (target column)
print('Correlations with Target (disease_progression)')
print(df_diabetes.corr()['disease_progression'].sort_values(ascending=False))


# In[7]:


# Check the column names in the DataFrame
print(df_diabetes.columns)


# # Split the label from the dataset

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split

# 'disease_progression' is the dependent variable (value to be predicted), so it will be

# removed from the training data and put into a separate DataFrame for labels.
label_column = 'disease_progression'

training_columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Split the training and test datasets and their labels.
X_train, X_test, y_train, y_test = train_test_split(df_diabetes[training_columns],
                                                    df_diabetes[label_column],
                                                    random_state=543)

# Compare the number of rows and columns in the original data to the training and test sets.
print(f'Original set:        {df_diabetes.shape}')
print('------------------------------')
print(f'Training features:   {X_train.shape}')
print(f'Test features:       {X_test.shape}')
print(f'Training labels:     {y_train.shape}')
print(f'Test labels:         {y_test.shape}')


# # Create a linear regression model

# In[9]:


# Construct a basic linear regression class object.

# Fit the training data to the regression object.
from sklearn.linear_model import LinearRegression

# Create a linear regression object
linear_regression = LinearRegression(fit_intercept=False)

# Fit the training data to the regression object
linear_regression.fit(X_train, y_train)

# Print the coefficients of the linear regression model
print("Coefficients:", linear_regression.coef_)


# # Compare the first ten predictions to actual values

# In[10]:


# Make predictions on the test set
y_pred = linear_regression.predict(X_test)

results_comparison = X_test.copy()
results_comparison['PredictedDiseaseProgression'] = np.round(y_pred, 2)
results_comparison['ActualDiseaseProgression'] = y_test.copy()

# View examples of the predictions compared to actual energy output.
results_comparison.head(10)


# # Calculate the error between predicted and actual values

# In[11]:


# Print the mean squared error (MSE) for the model's predictions on the test set.
# Import mean squared error function
from sklearn.metrics import mean_squared_error as mse

# Calculate the mean squared error (MSE) for the model's predictions on the test set
cost = mse(y_test, y_pred)
print('Cost (mean squared error): {:.2f}'.format(cost))


# # Plot lines of best fit for four features

# In[12]:


# Use Seaborn to create subplots for the four features that have the strongest correlation with the label.
# Also plot a line of best fit for each feature.

import seaborn as sns
import matplotlib.pyplot as plt

# Select the features with the strongest correlation with the "DiseaseProgression" label
strongest_corr_features = ['age', 'bmi', 's5', 's1']

# Create subplots for the four features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Define a color palette for scatter points
colors = ['blue', 'green', 'orange', 'purple']

# Plot lines of best fit for each feature
for i, feature in enumerate(strongest_corr_features):
    row = i // 2
    col = i % 2
    sns.regplot(x=X_test[feature], y=np.ravel(y_pred), scatter_kws={'color': colors[i]}, line_kws={'color': colors[i]}, ax=axes[row, col])
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Disease Progression')

plt.tight_layout()
plt.show()


# In[ ]:




