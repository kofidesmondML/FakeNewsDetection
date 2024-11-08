import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score


features_path='./data/ExtractedFeatures.csv'


def run_classification(X,y):

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

  precision_list=[]
  recall_list=[]
  f1_list=[]
  accuracy_list=[]
  auc_scores = []
  AVG_precision_scores = []


  for tr_ind, tst_ind in skf.split(X,y):
      print(tr_ind)
      X_train = X.iloc[tr_ind]
      X_test = X.iloc[tst_ind]
      y_train = y.iloc[tr_ind]
      y_test = y.iloc[tst_ind]

      #clf = LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs')
      clf = RandomForestClassifier(random_state=0,class_weight="balanced")
      clf.fit(X_train, y_train)  
      # predict the labels on test dataset
      predictions = clf.predict(X_test)

      proba = clf.predict_proba(X_test)[:,1]
      

      #evaluation
      precision = metrics.precision_score(y_test, predictions)
      recall = metrics.recall_score(y_test, predictions)
      f1 = metrics.f1_score(y_test, predictions)
      accuracy = metrics.accuracy_score(y_test,predictions)
      print(metrics.confusion_matrix(y_test,predictions))
      
      precision_list.append(precision)
      recall_list.append(recall)
      f1_list.append(f1)
      accuracy_list.append(accuracy)
      auc_scores.append(roc_auc_score(y_test, proba))
      AVG_precision_scores.append(average_precision_score(y_test, proba))


  print("precision = ", round(np.mean(precision_list)*100,3),"\n", 
        "recall = ",round(np.mean(recall_list)*100,3),"\n",
        "f1 = ",round(np.mean(f1_list)*100,3),"\n",
        "accuracy = ",round(np.mean(accuracy_list)*100,3),"\n", 
        "AUROC = ", round(np.mean(auc_scores)*100,3),"\n", 
        "Average Precision = ", round(np.mean(AVG_precision_scores)*100,3),"\n", )
  
features_df=pd.read_csv(features_path)
print(features_df['label'])
features_df['label'] = features_df['label'].apply(lambda x: 0 if "Real" in x else 1)
print(features_df['label'])
X = features_df.drop(columns=['label','Named Entities'])
y = features_df['label']
nan_columns = features_df.columns[features_df.isna().any()].tolist()
nan_counts = features_df.isna().sum()

print("Columns with NaN values:")
for col in nan_columns:
    print(f"{col}: {nan_counts[col]} NaNs")

run_classification(X,y)
  


  
