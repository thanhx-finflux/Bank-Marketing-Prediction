# %% [markdown]
# **Step 1: Loading the dataset**

# %%
# Loading data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

bank = pd.read_csv('bank marketing.csv')

# Data shape
print('Data shape:', bank.shape)
# Columns in the dataset
print('Columns in the dataset:\n',bank.columns.tolist())

# %% [markdown]
# **Step 2: Understanding the dataset**

# %%
# Summary of the dataset
print(bank.head())
print(bank.tail())
print('Data info:')
print(bank.info())
print('\nData description of numerical columns:')
print(bank.describe())

# %%
# Encoding categorical variables
bank['deposit'] = bank['deposit'].map({'no': 0, 'yes': 1})

# %%
# Target variable distribution
print('Target variable distribution:')
print(bank['deposit'].value_counts(normalize = True))

# %%
# Plot target variable distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'dark')
plt.figure(figsize = (8, 6))
sns.countplot(x = 'deposit', data = bank, palette = 'Set2')
plt.title('Target variable distribution')
plt.xlabel('Deposit')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# %%
# Categorical features unique values
cat_cols = bank.select_dtypes(include = ['object']).columns.tolist()
print('Categorical features and their unique values:')
for col in cat_cols:
    print(f'{col}: {bank[col].unique()}\n')

# %%
# Horizontal bar plot for categorical features
plt.figure(figsize = (24, 12))
for idx, col in enumerate(cat_cols):
    plt.subplot(2, 5, idx + 1)
    bank[col].value_counts(normalize = True).plot(kind = 'barh', color = 'skyblue')
    plt.title(col)
    plt.xlabel('Proportion')
    plt.ylabel('Categories')
plt.tight_layout()
plt.show()

# %%
# Boxplot for numerical features
plt.figure(figsize = (12, 6))
num_col = ['balance', 'age']
for col in num_col:
    sns.boxplot(data = bank[col], orient = 'h', color = 'lightblue')
    plt.title('Boxplot of numerical features')
    plt.xlabel('Values')
    plt.ylabel('Features')
    plt.show()

    


# %% [markdown]
# **Step 3: Preprocess Data**

# %%
# Encoding categorical variables
bank1 = bank.copy()
bank1[['default', 'housing', 'loan']] = bank1[['default', 'housing', 'loan']].apply(lambda x: x.map({'no': 0, 'yes': 1}))
dumm_cols = bank1.select_dtypes(include = ['object']).columns.tolist()
bank_dummies = pd.get_dummies(bank1[dumm_cols], drop_first = True).astype(int)
# Merging the dummy variables with the bank1 DataFrame
bank1= pd.concat([bank1, bank_dummies], axis = 1)
# Dropping the original categorical columns
bank1.drop(columns = dumm_cols, inplace =  True)

# %%
# Drop day and duration columns because the are not useful for prediction
bank1.drop(columns = ['day', 'duration'], inplace = True)

# %%

# Data shape after preprocessing
print('Data shape after preprocessing:', bank1.shape)
# Data info after preprcessing
print('Data info after preprocessing:')
print(bank1.info())

# %%
# Five rows of the preprocessed data
print('Five rows of the preprocessed data:')
print(bank1.head())

# %% [markdown]
# **Step 4: Split Data**

# %%
from sklearn.model_selection import train_test_split
# Slipting the data into training and testing sets
X= bank1.drop(columns= 'deposit')
y = bank1['deposit']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = y)
print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)
print('Training set target variable distribution:', 
      y_train.value_counts(normalize = True))
print('Testing set targer variable distribution:',
      y_test.value_counts(normalize = True))


# %%
# Scaling the balance and age columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit the scaler on the training data
X_train[['balance', 'age']] = scaler.fit_transform(X_train[['balance', 'age']])
# Transform the test data using the same scaler
X_test[['balance', 'age']] = scaler.transform(X_test[['balance', 'age']])


# %% [markdown]
# **Step 5: Build and Train Models**

# %%
# Create DataFrame to store the evaluation metrics: Model, Test F1-Score, Train F1-Score, F1-Score Gap, ROC-AUC Score, CV Score
evaluation_df = pd.DataFrame(columns=['Model', 
                                      'Test F1-Score', 
                                      'Train F1-Score', 
                                      'F1-Score Gap', 
                                      'ROC-AUC Score',
                                      'CV Score'])
# Model evaluation, including accuracy, precision, recall, and F1-score, overfitting, and underfitting
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    global evaluation_df
    # Predicting on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Evaluating the model
    # Training set evaluation
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)
    
    # Printing the evaluation metrics for training set
    print(f'{model_name} Train Model Evaluation:')
    print(f'Accuracy: {accuracy_train:.4f}')
    print(f'Precision: {precision_train:.4f}')
    print(f'Recall: {recall_train:.4f}')
    print(f'F1 Score: {f1_train:.4f}\n')

    # Testing set evaluation
    accurancy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_test_pred_proba)

    # Printing the evaluation metrics for testing set    
    print(f'{model_name} Test Model Evaluation:')
    print(f'Accuracy: {accurancy_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'F1 Score: {f1_test:.4f}\n')
    print(f'ROC AUC Score: {auc:.4f}\n')

    # Cross-validation for Logistic Regression
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(estimator=model,
                                X=X_train,
                                y=y_train,
                                cv=5,
                                scoring='accuracy')
    cv_scores_mean = cv_scores.mean()
    print(f'Mean cross-validation score: {cv_scores_mean:.4f}\n')
    
    # Append the evaluation_df
    new_row = pd.DataFrame([{'Model': model_name,
                            'Test F1-Score': f1_test,
                            'Train F1-Score': f1_train,
                            'F1-Score Gap': f1_train - f1_test,
                            'ROC-AUC Score': auc,
                            'CV Score': cv_scores_mean}])
    # Concatenate the new row to the evaluation_df if the model is not already present
    evaluation_df = pd.concat([evaluation_df, new_row], ignore_index=True)
# Model selection and evaluation
# Primary metric: F1-Score for balanced performance
# Secondary meytric: ROC-AUC Score for model descrimination
# Third F1-Score Gap to check for overfitting or underfitting
# Fourth CV Score to check for model stability
# Select the best model based on the highest F1-Score and ROC-AUC Score in test set and lowest F1-Score Gap
# Complexity of the model is also considered simple models are preferred over complex models if they perform similarly


# %% [markdown]
# **1. Logistic Regression**

# %% [markdown]
# **a. Initial Logistic Regression model**

# %%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter = 1000, random_state = 42)
# Fitting the model on the training data
log_reg.fit(X_train, y_train)

# %%
# Evaluation of the Logistic Regrssion initial model
evaluate_model(X_train, X_test, y_train, y_test, log_reg, 'Logistic Regression (Initial)')

# %% [markdown]
# **b. Grid Search for Logistic Regression**

# %%
# Hyperparameter tuning using Grid Search for Logistic Regression
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', 'none'] 
}
# Note: 'liblinear' supports 'l1' penalty, while 'saga' supports both 'l1' and 'l2' penalties.
# Performing Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator = log_reg,
                           param_grid = param_grid,
                          scoring = 'accuracy', 
                          cv = 5,
                          n_jobs = -1,
                          verbose = 1)
grid_search.fit(X_train, y_train)
# Best parameters from Grid Search
print('Best parameters from Grid Search:', grid_search.best_params_)
# Best score from Grid Search
print('Best score from Grid Search:', grid_search.best_score_)
# Best estimator from Grid Search
best_log_reg = grid_search.best_estimator_



# %%
# Evaluating the best Logistic Regression model from Grid Search
evaluate_model(X_train, X_test, y_train, y_test, best_log_reg, 'Logistic Regression (Grid Search)')

# %% [markdown]
# **c. LogistictRegressionCV**

# %%
# Logistic Regression cross-validation with hyperprameter tuning
# Using LogisticRegressionCV for hyperparameter tuning
from sklearn.linear_model import LogisticRegressionCV

# Using LogisticRegressionCV for hyperparameter tuning
log_reg_cv = LogisticRegressionCV(Cs = [0.01, 0.1, 1, 10, 100],
                                  penalty = 'l2',
                                  scoring = 'accuracy', 
                                  cv = 5, 
                                  n_jobs = -1,
                                  verbose = 1)
# Fitting the LogisticRegressionCV model on the training data
log_reg_cv.fit(X_train, y_train)
# Best parameters from LogisticRegressionCV
print('Best parameters from LogisticRegressionCV:', log_reg_cv.C_)
# Best score from LogisticRegressionCV
print('Best score from LogisticRegressionCV:', log_reg_cv.score(X_test, y_test))
# Best estimator from LogisticRegressionCV
best_log_reg_cv = log_reg_cv


# %%
# Evaluation of the Logistic Regression CV model
evaluate_model(X_train, X_test, y_train, y_test, best_log_reg_cv, 'Logistic Regression CV')

# %% [markdown]
# **2. Decision Tree**

# %% [markdown]
# **a. Initial Decision Tree Model**

# %%
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Initializing the Decsion Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state = 42)
# Fitting the Decision Tree Clssifier on the training data
dt_classifier.fit(X_train, y_train)


# %%
# Evaluating the Decision Tree Classifier initial model
evaluate_model(X_train, X_test, y_train, y_test, dt_classifier, 'Decision Tree Classifier(Initial)')

# %% [markdown]
# **b. Grid Search for Decision Tree Classifier**

# %%
# Hyperparameter tuning using Grid Search for Decision Tree Classifier
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', 'none']
}

# Performing Grid Search for hyperparameter tuning
grid_search_dt = GridSearchCV(estimator = dt_classifier,
                              param_grid = param_grid_dt,
                              scoring = 'accuracy',
                              cv = 5,
                              n_jobs = -1,
                              verbose = 1)
grid_search_dt.fit(X_train, y_train)
# Best parameters from Grid Search for Decision Tree Classifier
print('Best parameters from Grid Search for Decision Tree Classifier:')
print(grid_search_dt.best_params_)
# Best score from Grid Search for Decision Tree Classifier
print('Best score from Grid Search for Decision Tree Classifier:')
print(grid_search_dt.best_score_)
# Best estimator from Grid Search for Decision Tree Classifier
best_dt_classifier = grid_search_dt.best_estimator_

# %%
# Evaluating the best Decision Tree Classifier model with Grid Search
evaluate_model(X_train, X_test, y_train, y_test, best_dt_classifier, 'Decision Tree Classifier(Grid Search)')

# %% [markdown]
# **3. Random Forest Classifier**

# %% [markdown]
# **a. Initial Random Forest Classifier**

# %%
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Initializing the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state = 42)
# Fitting the Random Forest Classifier on the training data
rf_classifier.fit(X_train, y_train)

# %%
# Evaluating the Random Forest Classifier initial model
evaluate_model(X_train, X_test, y_train, y_test, rf_classifier, 'Random Forest (Initial)')

# %% [markdown]
# **b. RandomizedSearchCV**

# %%
# RandomizedSearchCV for Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV
# Hyperparameter tuning using Randomized Search for Random Forest Classifier
param_dist_rf = {
    'n_estimators': [50, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 10, 15, 20, 25, 30, 35], 
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10, 15], 
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', 'none']
}
# Performing Randomized Search for hyperparameter tuning
random_search_rf = RandomizedSearchCV(estimator = rf_classifier, 
                                       param_distributions = param_dist_rf,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 50,
                                       n_jobs = -1, 
                                       verbose = 1,
                                       random_state = 42)
# Fitting the Randomized Search model on the training data
random_search_rf.fit(X_train, y_train)
# Best parameters from Randomized Search for Random Forest Classifier
print('Best parameters from Randomized Search for Random Forest Classifier:')
print(random_search_rf.best_params_)
# Best score from Randomized Search for Random Forest Classifier
print('Best score from Randomized Search for Random Forest Classifier:')
print(random_search_rf.best_score_)
# Best estimator from Randomized Search for Random Forest Classifier
best_rf_classifier_random = random_search_rf.best_estimator_


# %%
# Evaluating the best Random Forest Classifier model from Randomized Search
evaluate_model(X_train, X_test, y_train, y_test, best_rf_classifier_random, 'Random Forest (Randomized Search)')

# %% [markdown]
# **4. XGBoost Classifier**

# %% [markdown]
# **a. Intial XGBoost Classifier**

# %%
# XGBoost Classifier
from xgboost import XGBClassifier
# Initializing the XGBoost Classifier
xgb_classifier = XGBClassifier(random_state = 42,
                               use_label_encoder = True, # Using label encoding for categorical features
                               eval_metric = 'logloss') # Evaluation metric for XGBoost
# Fitting the XGBoost Classifier on the training data
xgb_classifier.fit(X_train, y_train)

# %%
# Evaluating the XGBoost Classifier initial model
evaluate_model(X_train, X_test, y_train, y_test, xgb_classifier, 'XGBoost Classifier(Initial)')

# %% [markdown]
# **b. RandomizedSearchCV**

# %%
# Tuning hyperparameters using RandomizedSearchCV for XGBoost Classifier
from sklearn.model_selection import RandomizedSearchCV
# Hyperparameter tuning using Randomized Search for XGBoost Classifier
param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200], 
    'max_depth': [3, 5, 10, 15, 20], 
    'learning_rate': [0.01, 0.1, 0.2, 0.3], 
    'subsample': [0.5, 0.7, 1.0], 
    'colsample_bytree': [0.5, 0.7, 1.0], 
    'gamma': [0, 0.1, 0.2, 0.3], 
    'reg_alpha': [0, 0.1, 0.2], 
    'reg_lambda': [1e-5, 1e-4, 1e-3] 
}
# Performing Randomized Search for hyperparameter tuning
random_search_xgb = RandomizedSearchCV(estimator = xgb_classifier,
                                        param_distributions = param_dist_xgb, 
                                        scoring = 'accuracy', 
                                        cv = 5,
                                        n_iter = 50,
                                        n_jobs = -1,
                                        verbose = 1,
                                        random_state = 42)
# Fitting the Randomized Search model on the training data
random_search_xgb.fit(X_train, y_train)
# Best parameters from Randomized Search for XGBoost Classifier
print('Best parameters from Randomized Search for XGBoost Classifier:')
print(random_search_xgb.best_params_)
# Best score from Randomized Search for XGBoost Classifier
print('Best score from Randomized Search for XGBoost Classifier:')
print(random_search_xgb.best_score_)
# Best estimator from Randomized Search for XGBoost Classifier
best_xgb_classifier_random = random_search_xgb.best_estimator_


# %%
# Evaluatiing the best XGBoost Classifier model from Randomized Search
evaluate_model(X_train, X_test, y_train, y_test, best_xgb_classifier_random, 'XGBoost Classifier(Randomized Search)')

# %% [markdown]
# **Step 6: Model Selection**
# 

# %%
# Evaluating the models based on the evaluation_df
print('Model Evaluation Summary:')
print(evaluation_df)

# %%
# Plot models comparison based on F1-Score
plt.figure(figsize = (12, 6))
sns.barplot(x = 'Model', 
            y = 'Test F1-Score', 
            data = evaluation_df, 
            palette = 'Set3')
plt.title('Models Comparison based on F1-Score')
plt.xlabel('Models')
plt.ylabel('Test F1-Score')
plt.xticks(rotation = 45)
# Display the F1-Score values on top of the bars
for index, value in enumerate(evaluation_df['Test F1-Score']):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')
plt.show()

# %%
# Option 1: Rank the models by weighted criteria
evaluation_df['Weighted Score'] = (evaluation_df['Test F1-Score'] * 0.4 +
                                   evaluation_df['ROC-AUC Score'] * 0.3 +
                                   (1 - evaluation_df['F1-Score Gap']) * 0.2 +
                                   evaluation_df['CV Score'] * 0.1)

# %%
# Plot comparison of models based on weighted score
plt.figure(figsize = (12, 6))
sns.barplot(x = 'Model', 
            y = 'Weighted Score',
            data = evaluation_df,
            palette = 'Set2')
plt.title('Models Comparison based on Weighted Score')
plt.xlabel('Models')
plt.ylabel('Weighted Score')
plt.xticks(rotation = 45)
# Display the Weighted Score values on top of the bars
for index, value in enumerate(evaluation_df['Weighted Score']):
    plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')
plt.show()


# %%
top_models = evaluation_df.sort_values(by = 'Weighted Score', ascending = False)
print('Top models based on weighted criteria:')
print(top_models.head(3))

# %%
# Option 2: Difine the threshold
best_model = evaluation_df[(evaluation_df['Test F1-Score'] >= evaluation_df['Test F1-Score'].quantile(0.5)) &
                           (evaluation_df['ROC-AUC Score'] >= evaluation_df['ROC-AUC Score'].quantile(0.5)) &
                           (evaluation_df['F1-Score Gap'] <= evaluation_df['F1-Score Gap'].quantile(0.5)) &
                           (evaluation_df['CV Score'] >= evaluation_df['CV Score'].quantile(0.5))]
print('Best model based on weighted criteria:')
print(best_model)

# %%
# Option 3: Display Top 3 models per metric
top_f1_models = evaluation_df.sort_values(by = 'Test F1-Score', ascending = False).head(3)
top_auc_models = evaluation_df.sort_values(by = 'ROC-AUC Score', ascending = False).head(3)
top_gap_models = evaluation_df.sort_values(by = 'F1-Score Gap', ascending = True).head(3)
top_cv_models = evaluation_df.sort_values(by = 'CV Score', ascending = False).head(3)


# %%
top_models = pd.concat([top_f1_models, top_auc_models, top_gap_models, top_cv_models])
most_common_model = top_models['Model'].value_counts()
print('Most common model across top metrics:')
print(most_common_model)

# %%
# Plot the top models based on count of occurrences
plt.figure(figsize = (12, 6))
most_common_model.plot(kind = 'bar', color = 'skyblue')
plt.title('Top Models Based on Count of Occurrences')
plt.xlabel('Models')
plt.ylabel('Count')
plt.xticks(rotation = 45)
plt.show()

# %%
# Option 4: Highest test F1-Score, ROC-AUC Score > 0.70, and lowest F1-Score Gap < 0.10
best_model = evaluation_df[(evaluation_df['Test F1-Score'] >= evaluation_df['Test F1-Score'].quantile(0.75)) &
                           (evaluation_df['ROC-AUC Score'] > 0.70) &
                           (evaluation_df['F1-Score Gap'] < 0.10)]
print('Best model based on highest F1-Score, ROC-AUC Score > 0.70, and lowest F1-Score Gap < 0.10:')
print(best_model)


# %%
# Based on the 3 options, the best model can be selected.
# The best model can be selected based on the weighted score, threshold criteria, or top models per metric.
# The final selection depends on the business requirements and the trade-offs between precision, recall, and F1-score.


