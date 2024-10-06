#Fine turning your model
#instructions
'''Import confusion_matrix and classification_report.
Fit the model to the training data.
Predict the labels of the test set, storing the results as y_pred.
Compute and print the confusion matrix and classification report for the test labels versus the predicted labels.'''

# Import confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Logistic regressison
#instructions
'''Import LogisticRegression.
Instantiate a logistic regression model, logreg.
Fit the model to the training data.
Predict the probabilities of each individual in the test set having a diabetes diagnosis, storing the array of positive probabilities as y_pred_probs.'''

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])


#instructions
'''Import roc_curve.
Calculate the ROC curve values, using y_test and y_pred_probs, and unpacking the results into fpr, tpr, and thresholds.
Plot true positive rate against false positive rate.'''

# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()



#instructions
'''Import roc_auc_score.
Calculate and print the ROC AUC score, passing the test labels and the predicted positive class probabilities.
Calculate and print the confusion matrix.
Call classification_report().'''


# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Calculate roc_auc_score
print(roc_auc_score(y_test,y_pred_probs))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test,y_pred))


#Hyperparameter tuning
#instructions
'''Import GridSearchCV.
Set up a parameter grid for "alpha", using np.linspace() to create 20 evenly spaced values ranging from 0.00001 to 1.
Call GridSearchCV(), passing lasso, the parameter grid, and setting cv equal to kf.
Fit the grid search object to the training data to perform a cross-validated grid search.'''

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso,param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train,y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))


#instructions
'''Create params, adding "l1" and "l2" as penalty values, setting C to a range of 50 float values between 0.1 and 1.0, and class_weight to either "balanced" or a dictionary containing 0:0.8, 1:0.2.
Create the Randomized Search CV object, passing the model and the parameters, and setting cv equal to kf.
Fit logreg_cv to the training data.
Print the model's best parameters and accuracy score.'''

# Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(logreg_cv.best_score_))



#Preparing data
#instructions
'''Use a relevant function, passing the entire music_df DataFrame, to create music_dummies, dropping the first binary column.
Print the shape of music_dummies.'''

# Create music_dummies
music_dummies = pd.get_dummies(music_df, drop_first = True)

# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))


#instructions
'''Create X, containing all features in music_dummies, and y, consisting of the "popularity" column, respectively.
Instantiate a ridge regression model, setting alpha equal to 0.2.
Perform cross-validation on X and y using the ridge model, setting cv equal to kf, and using negative mean squared error as the scoring metric.
Print the RMSE values by converting negative scores to positive and taking the square root.'''

# Create X and y
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies['popularity'].values

# Instantiate a ridge model
ridge = Ridge(alpha=0.2)

# Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

# Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))


#Handling Missing Data
#instructions
'''Print the number of missing values for each column in the music_df dataset, sorted in ascending order.
Remove values for all columns with 50 or fewer missing values.
Convert music_df["genre"] to values of 1 if the row contains "Rock", otherwise change the value to 0.'''

# Print missing values for each column
print(music_df.isna().sum().sort_values())

# Remove values where less than 5% are missing
music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])

# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)

print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))



#instructions
'''Import SimpleImputer and Pipeline.
Instantiate an imputer.
Instantiate a KNN classifier with three neighbors.
Create steps, a list of tuples containing the imputer variable you created, called "imputer", followed by the knn model you created, called "knn".'''

# Import modules
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors = 3)

# Build steps for the pipeline
steps = [("imputer", imputer), 
         ("knn", knn)]



#instructions
'''Create a pipeline using the steps you previously defined.
Fit the pipeline to the training data.
Make predictions on the test set.
Calculate and print the confusion matrix.'''

steps = [("imputer", imp_mean),
        ("knn", knn)]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))


#Centering and scaling
#instructions
'''Import StandardScaler.
Create the steps for the pipeline object, a StandardScaler object called "scaler", and a lasso model called "lasso" with alpha set to 0.5.
Instantiate a pipeline with steps to scale and build a lasso regression model.
Calculate the R-squared value on the test data.'''

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
print(pipeline.score(X_test, y_test))



#instructions
'''Build the steps for the pipeline: a StandardScaler() object named "scaler", and a logistic regression model named "logreg".
Create the parameters, searching 20 equally spaced float values ranging from 0.001 to 1.0 for the logistic regression model's C hyperparameter within the pipeline.
Instantiate the grid search object.
Fit the grid search object to the training data.'''

# Build the steps
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001,1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid= parameters)

# Fit to the training data
cv.fit(X_train,y_train)
print(cv.best_score_, "\n", cv.best_params_)


#Evaluating multiple models
#instructions
'''Write a for loop using model as the iterator, and model.values() as the iterable.
Perform cross-validation on the training features and the training target array using the model, setting cv equal to the KFold object.
Append the model's cross-validation scores to the results list.
Create a box plot displaying the results, with the x-axis labels as the names of the models.'''

models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv= kf)
  
  # Append the results
  results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels= models.keys())
plt.show()



#instructions
'''Import mean_squared_error.
Fit the model to the scaled training features and the training labels.
Make predictions using the scaled test features.
Calculate RMSE by passing the test set labels and the predicted labels.'''

# Import mean_squared_error
from sklearn.metrics import mean_squared_error

for name, model in models.items():
  
  # Fit the model to the training data
  model.fit(X_train_scaled,y_train)
  
  # Make predictions on the test set
  
  y_pred = model.predict(X_test_scaled)

  # Calculate the test_rmse
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("{} Test Set RMSE: {}".format(name, test_rmse))



#instructions
'''Create a dictionary of "Logistic Regression", "KNN", and "Decision Tree Classifier", setting the dictionary's values to a call of each model.
Loop through the values in models.
Instantiate a KFold object to perform 6 splits, setting shuffle to True and random_state to 12.
Perform cross-validation using the model, the scaled training features, the target training set, and setting cv equal to kf.'''

# Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():
  
  # Instantiate a KFold object
  kf = KFold(n_splits= 6, random_state= 12, shuffle= True)
  
  # Perform cross-validation
  cv_results = cross_val_score(model,X_train_scaled, y_train, cv= kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()


#instructions
'''Create the steps for the pipeline by calling a simple imputer, a standard scaler, and a logistic regression model.
Create a pipeline object, and pass the steps variable.
Instantiate a grid search object to perform cross-validation using the pipeline and the parameters.
Print the best parameters and compute and print the test set accuracy score for the grid search object.'''

# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))


