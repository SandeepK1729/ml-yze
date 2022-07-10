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
gnb = GaussianNB()
svm = SVC()
dt = tree.DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators = 25)
ada = AdaBoostClassifier()

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def entry_point():
    global df,x,y,X_train, X_test,y_train,y_test
    cm=[]
    accuracy=-1
    precisons=-1
    kval=-1
    classifier="not set"
    dataset="not set"
    plot_url=""
    if request.method=='POST':
        classifier=request.form['classifier']
        dataset=request.form['dataset']
        kval=request.form['kval']
        if  dataset=="diabetes":
            df = pd.read_csv("csvfiles/diabetes.csv")
            y = df['Outcome']
            x = df.drop("Outcome",axis = 1)
            X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
        elif dataset=="strokes":
            df = pd.read_csv("csvfiles/strokes.csv")
            y = df['stroke']
            x = df.drop("stroke",axis = 1)
            X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
        elif dataset=="wine":
            df = datasets.load_wine()
            y = df.target
            x = df.data
            X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
        if classifier=="naivebayes":
            gnb.fit(X_train,y_train)
            y_pred_nb = gnb.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred_nb)
            precisons=precision_score(y_test,y_pred_nb,average='weighted')
            cm=confusion_matrix(y_test,y_pred_nb)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            #print(accuracy)
            #print(precisons)
            #print(cm)
        elif classifier=="svm":
            svm.fit(X_train,y_train)
            y_pred_svm=svm.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred_svm)
            precisons=precision_score(y_test,y_pred_svm,average='weighted')
            cm=confusion_matrix(y_test,y_pred_svm)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        elif classifier=="decisiontree":
            dt.fit(X_train,y_train)
            y_pred_dt=dt.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred_dt)
            precisons=precision_score(y_test,y_pred_dt,average='weighted')
            cm=confusion_matrix(y_test,y_pred_dt)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        elif classifier=="randomforest":
            rf.fit(X_train,y_train)
            y_pred_rf = rf.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred_rf)
            precisons=precision_score(y_test,y_pred_rf,average='weighted')
            cm=confusion_matrix(y_test,y_pred_rf)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        elif classifier=="adaboost":
            ada.fit(X_train,y_train)
            y_pred_ada=ada.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred_ada)
            precisons=precision_score(y_test,y_pred_ada,average='weighted')
            cm=confusion_matrix(y_test,y_pred_ada)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        elif classifier=="knn":
            knnM=KNeighborsClassifier(n_neighbors=int(kval))
            knnM.fit(X_train,y_train)
            y_pred_knn = knnM.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred_knn)
            precisons=precision_score(y_test,y_pred_knn,average='weighted')
            cm=confusion_matrix(y_test,y_pred_knn)
            heatmap = sns.heatmap(cm, annot=True, cmap="Blues")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Heapmap for confusion matrix')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('analyze.html',resk=kval,resacc=(accuracy*100),respre=(precisons),rescm=cm,resclass=classifier,resdata=dataset,plot_url=plot_url)

if __name__ == '__main__':
    app.run()