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
df = pd.read_csv("C:\\Users\\Will-Meister\\Desktop\\Projects\\MachineLearningIntro\\CourseRepoFiles\\MachineLearningWithPython-master\\Notebooks\\data\\pima-data.csv")

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