import numpy as np
import pandas as pd
from collections import Counter
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_curve, PrecisionRecallDisplay

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']     #amino acid symbols to assist in feature generation, 'X'- the unknown amino acid(s) is also included.
train_data_path = input("Enter train data csv file path:")                                                                  #prompt user to input the paths of training and testing csv files
test_data_path = input("Enter test data csv file path:")

train_data = pd.read_csv(train_data_path)           #open the train and test data csv files as specified by the user entered paths
test_data = pd.read_csv(test_data_path)
train_data_sequences = train_data['Sequence']       #extract the amino acid residue sequences from the training data
train_data_labels = train_data['label']             #extract the class labels (0,1) corresponding to the amino acid residue sequences from the training data
test_data_sequences = test_data['Sequence']         #extract the amino acid residue sequences from the testing data

#-----------------------------------------------------------------------------------------------------------------#

#feature generation using one hot encoding
def one_hot_encode(seq):
  ''' Uses the 21 characters in the amino_acids array and creates a 2D matrix of one hot encoding of the input sequence
  In our use case, the sequences have a fixed length of 17, the 2D matrix will thus be 17x21 The matrix is then flattened to one dimension.
  Input: amino-acid residue sequence string
  Return: a flattened array of dimension 1 x (length of input residue*21)
  '''
  absent_aa = list(set(amino_acids) - set(seq))                                          #get amino acids not present in the input sequence
  s = pd.DataFrame(list(seq)) 
  x = pd.DataFrame(np.zeros((len(seq),len(absent_aa)),dtype=int),columns=absent_aa)      #fill the positions of absent amino acids with 0s
  matrix2d = s[0].str.get_dummies(sep=',')                                               #fill positions present amino acids with 1s and join the the 0s and 1s matrix          
  matrix2d = matrix2d.join(x)
  matrix2d = matrix2d.sort_index(axis=1)
  matrixflattened = matrix2d.values.flatten()                                             #flatten the 2D matrix to a 1D array
  return matrixflattened

#-----------------------------------------------------------------------------------------------------------------#

def machine_learning_model():
  ''' Uses an ensemble of Random Forest Classifiers using EasyEnsembleClassifier from imblearn which combines bagging and random forest as base estimator for imbalanced data.
  Return: the model variable.
  '''
  model = EasyEnsembleClassifier(n_estimators=10, random_state=0, base_estimator=RandomForestClassifier() ,sampling_strategy='auto')     #(hypertuned) ensemble of 10 random forest classifiers
  return model

#-----------------------------------------------------------------------------------------------------------------#

def pred_machine_learning_model(trainX, validationX, trainy, validationy, train_data_onehot, train_data_labels):
  ''' Gets the machine learning model and preforms kfold cross validation on the model with roc_auc as scoring criteria. Then uses the model to make predictions on test dataset.
  Input: the training features , training labels and validation features, validation labels split from the training datset, the original training features, training labels and test features as separate dataframes.
  Return: the final refitted model
  '''
  model = machine_learning_model()                                              #gets the machine learning model variable
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)         #stratified K-Fold 5 times with different randomization in each repetition. K=10. 
  scores = cross_val_score(model, trainX, trainy, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
  print("scores from k-fold cross validation: \n", scores)
  model.fit(trainX, trainy)                                                     #fits the model on the splitted training data
  valpred = model.predict(validationX)
  print("\nValidation predictions report: ")
  print(classification_report(validationy, valpred))                           #scores, reports, confusion matrix of the validation datset predeicted
  print(confusion_matrix(validationy, valpred))
  print("\nConfusion Matrix for Validation Set: ")
  model.fit(train_data_onehot, train_data_labels)                               #refit the model on the entire training dataset
  return model

#-----------------------------------------------------------------------------------------------------------------#

def main():
  train_data_onehot = train_data_sequences.apply(lambda x: pd.Series(one_hot_encode(x)),1)                   #one hot encode the sequences in training dataset (feature generation)
  test_data_onehot = test_data_sequences.apply(lambda x: pd.Series(one_hot_encode(x)),1)                      #one hot encode the sequences in testing dataset (feature generation)
  trainX, testX, trainy, testy = train_test_split(train_data_onehot, train_data_labels, test_size=0.2, random_state=2)     #split the dataset into training and testing data to feed into the ML model. The split test dataset constitutes 20% of the original training dataset
  model = pred_machine_learning_model(trainX, testX, trainy, testy, train_data_onehot, train_data_labels)
  predictions = model.predict(test_data_onehot)                                                               #use the refitted model to make predictions on the final testing data
  print("\nNumber of 1s and 0s in the predicted labels for test dataset: ")
  print(Counter(predictions))        
  lat_arr = np. reshape(predictions, 9582)
  finalpredictions = pd.DataFrame({'ID': np.array(test_data['ID']) ,'Label':lat_arr}, index=list(range(10001, 19583)))     #create the final predictions dataframe and export it as a csv.
  finalpredictions.to_csv('Group_18_Predictions.csv', index=False)
  
main()