"""
Data Description
****************
pima-data.csv
Details:
 - Female patients at least 21 years old
 - 768 patient observation rows
 - 10 columns
    - 9 feature columns: # pregnancies, blood pressure, glucose level, insulin level, etc.
    - 1 class column: Diabetes -> True or False
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load and review data
df = pd.read_csv("pima-data.csv")

#Check structure of data (dimensions)
print(df.shape)

#Check first X amount of rows of dataframe
print(df.head(5))

#Check last X amount of rows of dataframe
print(df.tail(5))

#Check for null values in dataframe, returns false if no nulls
print(df.isnull().values.any())

#Check for correlations
def plot_correlations(df, size=11):
    """
    Plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input: 
        df: pandas dataframe
        size: vertical and horizontal dimensions of the plot
    
    Displays:
        matrix of correlation between columns
            Blue-cyan-yellow-red-darkred -> the order of correlation from least to most correlated
            yellow-green-cyan-blue-purple -> the updated version
            Expect a dark red line running from top left to bottom right
    """
    correlation = df.corr() #data frame for correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(correlation) #color codes rectangles by correlation value
    plt.xticks(range(len(correlation.columns)), correlation.columns) #draw x tick marks
    plt.yticks(range(len(correlation.columns)), correlation.columns) #draw y tick marks


#plot might not work in vsc, it works in jupyter notebook 
plot_correlations(df)

#show table of correlations, closer to 1 the more correlated
print(df.corr())
#NOTE: if there is a direct correlation, such as thickness & skin where it's corr = 1, remove one of the two columns as to not duplicate weighting

#remove skin column
del df["skin"]

#replace boolean cells with integers (True = 1, False = 0)
#to do this we need a map to assign the old vals to new ones, which we input into the map function within pandas in the assignment below
tf_map = {True: 1, False: 0}
df["diabetes"] = df["diabetes"].map(tf_map)

#Check true/false ratio of diabetes cases
num_true = len(df.loc[df["diabetes"] == True])
num_false = len(df.loc[df["diabetes"] == False])
print("Number of true cases: {0} ({1:2.2f}%)".format(num_true, (num_true/(num_true + num_false) * 100)))
print("Number of false cases: {0} ({1:2.2f}%)".format(num_false, (num_false/(num_true + num_false) * 100)))

"""
Splitting the data: 70% for training, 30% for testing
"""

from sklearn.model_selection import train_test_split

feature_col_names = ["num_preg","glucose_conc","diastolic_bp","thickness","insulin","bmi","diab_pred","age"]
predicted_class_names = ["diabetes"]

X = df[feature_col_names].values        #predictor feature columns (8 X m)
y = df[predicted_class_names].values    #predicted class(1=true, 0=false) column (1 X m)
split_test_size = 0.30                  #test size is 30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state=42) #any number can be used for random_state, it must be consistent though

#Verify there is a 70/30% split in data for training and testing
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index))*100))
print("{0:0.2f}% in training set".format((len(X_test)/len(df.index))*100))

#Verify predicted values were split correctly
print("Original True : {0} ({1:0.2f}%)".format(len(df.loc[df["diabetes"] == 1]), (len(df.loc[df["diabetes"] == 1])/len(df.index))*100))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df["diabetes"] == 0]), (len(df.loc[df["diabetes"] == 0])/len(df.index))*100))
print("")
print("Training True : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train))*100))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train))*100))
print("")
print("Test True : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test))*100))
print("Test False : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test))*100))

#deal with null values
#thickness has values of 0 in some cases, thats not physically possible (research to the best of your ability to check if vals can be null)
#check # of rows with missing data/null values
print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc {0}".format(len(df.loc[df["glucose_conc"] == 0])))
print("# rows missing diastolic_bp {0}".format(len(df.loc[df["diastolic_bp"] == 0])))
print("# rows missing thickness {0}".format(len(df.loc[df["thickness"] == 0])))
print("# rows missing insulin {0}".format(len(df.loc[df["insulin"] == 0])))
print("# rows missing bmi {0}".format(len(df.loc[df["bmi"] == 0])))
print("# rows missing diab_pred {0}".format(len(df.loc[df["diab_pred"] == 0])))
print("# rows missing age {0}".format(len(df.loc[df["age"] == 0])))
print("")

#for features missing in a large number of rows, try imputing
#you could replace with mean, median
#or replace with expert knowledge derived value

#Impute with mean
from sklearn.impute import SimpleImputer
#impute with mean all 0 readings
fill_zeros = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_zeros.fit_transform(X_train)
X_test = fill_zeros.fit_transform(X_test)


#implement training algorithm - Naive Bayes
from sklearn.naive_bayes import GaussianNB

#create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

#predict values using the training data
nb_predict_train = nb_model.predict(X_train)
nb_predict_test = nb_model.predict(X_test)

#import the performance metrics library
from sklearn import metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train))) #test training data
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test))) #test testing data
print("")

#Review Confusion Matrix
print("**Confusion Matrix**")
#left columns in printed display matrix are predicted false, right columns are predicted true
#the rows are the actual values, top row is actual false, bottom is actual true
""" 
Example: TN = true negative (actual not diabetes, predicted not diabetes) |  FP = false positive (actual not diabetes, predicted to be diabetes) 
         FN = false negative (actual diabetes, predicted not diabetes)    |  TP = true positive (actual diabetes, predicted to be diabetes)

TN FP
FN TP

"""
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("**Classification Report**")
print(metrics.classification_report(y_test, nb_predict_test))

#Switching to random forest algorithm to improve performance
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42) #create random forest object
rf_model.fit(X_train, y_train.ravel())

rf_predict_train = rf_model.predict(X_train)
rf_predict_test = rf_model.predict(X_test)

#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train))) #measure accuracy of training data
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test))) #measure accuracy of test data
#the differences between the training data accuracy and test data accuracy is too large, this means the model has learned the training data too well
#the above problem is known as OVERFITTING
#**Solutions to overfitting below**
#regularization hyperparameters can be used to reduce the accuracy on trained data, but increase it on outside data
#cross validation
#Sacrifice some training perfection to improve overall performance: this is known as the the bias-variance trade off

print("**Classification Report**")
print(metrics.classification_report(y_test, rf_predict_test))


#switching to logistic regression, it works well in classification scenarios, and is simpler than radom forest
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test))) #measure accuracy of training data

print("**Classification Report**")
print(metrics.classification_report(y_test, lr_predict_test))
#recall score is still too low

#class imbalances can influence results, if there aren't enough positive results in the data set or there are too many even
#Add a balance weight hyperparameter


#loop through training to find best C value
#Add in class_weight = "balanced" hyperparameter to account for influenced results
C_start = 0.1
C_end = 5
C_inc = 0.1
C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight = "balanced", random_state=42)
    lr_model_loop.fit(X_train,y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc
                       
#matplotlib visualization, works in jupyter

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("First max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")

#now that we found the best C value, replace in C

lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, random_state = 42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

print("**Classification Report**")
print(metrics.classification_report(y_test, lr_predict_test))

#Recall value is 0.71 for True (1) now, this passes the minimum 70% set in my question
#still needs tuning, as the regularization value was balanced based on the test data
#try cross validation, specifically K-fold Cross validation
    #this splits the training data into a number of folds, with one assigned to validate, each fold is used to validate one at a time

    #psuedo code for this

    #for each fold
        #determine best hyperparameter value
    
    #Set model hyperparameter value to average best

#alg + Cross validation = algCV (naming convention when using cross validation)

#**Logistic Regression with cross validation**
from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs = 3, cv = 10, refit = False, class_weight="balanced")
"""
n_jobs = number cores to use in our system
Cs = the number of values it will try, trying to find the best value for the regularization parameter for each fold
cv = number of folds
"""
lr_cv_model.fit(X_train, y_train.ravel())
lr_cv_predict_test = lr_cv_model.predict(X_test)

#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test))) #measure accuracy of training data
print(metrics.confusion_matrix(y_test, lr_cv_predict_test))
print("")
print("**Classification Report**")
print(metrics.classification_report(y_test, lr_cv_predict_test))

#the cross validation drops our recall score on test data, but likely improves scores on real world data
