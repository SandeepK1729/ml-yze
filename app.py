from email import message
from django.shortcuts import render
from flask import Flask, render_template,request
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io


#model imports
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score,confusion_matrix

app=Flask(__name__)

#model initialization
dataset_names = ["Wine", "Diabetes", "Stroke"]
classifiers_list = {
    "naivebayes" : "Naive Bayes", 
    "svm" : "SVM",
    "decisiontree" : "Decision Tree Classifier",
    "randomforest" : "Random Forest Classifier",
    #(n_estimators = 25), 
    "adaboost" : "Adaboost Classifier",
    "knn" : "K Nearest Neighbor Classifier",
} # classifier key : name
clf = {
    "naivebayes" : GaussianNB, 
    "svm" : SVC,
    "decisiontree" : tree.DecisionTreeClassifier,
    "randomforest" : RandomForestClassifier,
    #(n_estimators = 25), 
    "adaboost" : AdaBoostClassifier,
    "knn" : KNeighborsClassifier,
} # classifier key : method address

def get_X_y(dataset) : 
    if  dataset=="Diabetes":
        df = pd.read_csv("csvfiles/diabetes.csv")
        y = df['Outcome']
        X = df.drop("Outcome",axis = 1)
        return (X, y)
    elif dataset=="Stroke":
        df = pd.read_csv("csvfiles/strokes.csv")
        y = df['stroke']
        X = df.drop("stroke",axis = 1)
        return (X, y)
    elif dataset=="Wine":
        X, y = datasets.load_wine(return_X_y=True)
        return (X, y)
    else : 
        return False

@app.route('/', methods = ["GET", "POST"])
def hello_world():
    result = {} 
    global X, y, X_train, X_test, y_train, y_test 

    if request.method=='POST':    
        
        classifier = request.form['classifier']
        dataset = request.form['dataset']
        kval = request.form['kval']
        

        if (dataset not in dataset_names) or (classifier not in clf ): 
            if dataset == "select" : 
                message = "Please Select the dataset"
            elif classifier == "select" : 
                message = "Please Select the Classifier"
            else :         
                message = "Classifier or dataset is not exist"
            return render_template("index.html", message = message, datasets = datasets, clfs = classifiers_list) 

        re = get_X_y(dataset)
        X, y = re[0], re[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    
        model_method = clf[classifier]
        
        if classifier == "knn": 
            model = model_method(n_neighbors = int(kval)) 
            result["kval"] = kval 
        elif classifier == "randomforest" : 
            model = model_method(n_estimators = 25) 
        else : 
            model = model_method()
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        result["accuracy"]  = accuracy_score(y_test,y_pred ) * 100
        result["precision"] = precision_score(y_test,y_pred,average='weighted')
        cm                  = confusion_matrix(y_test,y_pred)

        heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Heapmap for confusion matrix')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()

        img.seek(0)
        result["plot_url"] = base64.b64encode(img.getvalue()).decode('utf8')
        result["cm"] = cm 

        result["classifier"] = classifier
        result["dataset"] = dataset

    return render_template('index.html', datasets = dataset_names, clfs = classifiers_list, re = result)

if __name__ == '__main__':
    app.debug = True
    app.run()