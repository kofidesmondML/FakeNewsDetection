import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

features_path = './data/ExtractedFeatures.csv'
features_df = pd.read_csv(features_path)
features_df['label'] = features_df['label'].apply(lambda x: 0 if "Real" in x else 1)

X = features_df.drop(columns=['NewsID','label', 'Named Entities'])
y = features_df['label']

def run_classification(X, y):
    classifiers = {
        "Logistic Regression": LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs'),
        "Random Forest": RandomForestClassifier(random_state=0, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=0, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    metrics_dict = {
        "Classifier": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Accuracy": [],
        "AUROC": [],
        "Average Precision": []
    }

    for name, clf in classifiers.items():
        precision_list, recall_list, f1_list, accuracy_list, auc_scores, avg_precision_scores = [], [], [], [], [], []

        for tr_ind, tst_ind in skf.split(X, y):
            X_train, X_test = X.iloc[tr_ind], X.iloc[tst_ind]
            y_train, y_test = y.iloc[tr_ind], y.iloc[tst_ind]

            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

            precision_list.append(metrics.precision_score(y_test, predictions))
            recall_list.append(metrics.recall_score(y_test, predictions))
            f1_list.append(metrics.f1_score(y_test, predictions))
            accuracy_list.append(metrics.accuracy_score(y_test, predictions))
            auc_scores.append(roc_auc_score(y_test, proba))
            avg_precision_scores.append(average_precision_score(y_test, proba))

            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.savefig(f"results/confusion_matrix_{name}.png")
            plt.close()

        metrics_dict["Classifier"].append(name)
        metrics_dict["Precision"].append(np.mean(precision_list))
        metrics_dict["Recall"].append(np.mean(recall_list))
        metrics_dict["F1 Score"].append(np.mean(f1_list))
        metrics_dict["Accuracy"].append(np.mean(accuracy_list))
        metrics_dict["AUROC"].append(np.mean(auc_scores))
        metrics_dict["Average Precision"].append(np.mean(avg_precision_scores))

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv("results/classifier_metrics.csv", index=False)
    
    metrics_df.set_index("Classifier", inplace=True)
    metrics_df.T.plot(kind="bar", figsize=(12, 8))
    plt.title("Classifier Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/classifier_performance_metrics.png")
    plt.show()

run_classification(X, y)
