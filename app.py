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

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def entry_point():
    global df,X,y,X_train, X_test,y_train,y_test
    result = {
        cm         : [],
        accuracy   : -1,
        precison   : -1,
        kval       : -1,
        classifier : "not set",
        dataset    : "not set",
        plot_url   : "", 
    } 
    
    if request.method=='POST':
        classifier = request.form['classifier']
        dataset = request.form['dataset']
        kval = request.form['kval']
        
        if  dataset=="diabetes":
            df = pd.read_csv("csvfiles/diabetes.csv")
            y = df['Outcome']
            X = df.drop("Outcome",axis = 1)
        elif dataset=="strokes":
            df = pd.read_csv("csvfiles/strokes.csv")
            y = df['stroke']
            X = df.drop("stroke",axis = 1)
        elif dataset=="wine":
            df = datasets.load_wine()
            y = df.target
            X = df.data
            
        X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
        
        if classifier in clf: 
            model_method = clf[classifier]
            
            if classifier == "knn": 
                model = model_method(n_neighbors = int(kval)) 
                result[kval] = kval 
            elif classifier == "randomforest" : 
                model = model_method(n_estimators = 25) 
            else : 
                model = model_method()
            
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            result[accuracy] = accuracy_score(y_test,y_pred ) * 100
            result[precison]=precision_score(y_test,y_pred,average='weighted')
            result[cm] =confusion_matrix(y_test,y_pred)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        
        result[classifier] = classifier
        result[dataset] = dataset
    return render_template('analyze.html',re = result, clf = classifiers_list)

if __name__ == '__main__':
    app.run()