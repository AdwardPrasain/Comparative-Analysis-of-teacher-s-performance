from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

modeldtree=pickle.load(open('presentdtree.pkl','rb'));
modelrfc=pickle.load(open('presentrfc.pkl','rb'));
modellogreg=pickle.load(open('presentlogreg.pkl','rb'));
modelsvm=pickle.load(open('presentsvm.pkl','rb'));

modelpdtree=pickle.load(open('performancedtree.pkl','rb'));
modelprfc=pickle.load(open('performancerfc.pkl','rb'));
modelplogreg=pickle.load(open('performancelogreg.pkl','rb'));
modelpsvm=pickle.load(open('performancesvm.pkl','rb'));

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predictTeacher():
    return render_template("predict_teacher.html")

@app.route('/evaluate')
def performanceTeacher():
    return render_template("evaluate_teacher.html")

@app.route('/about')
def contactUs():
    return render_template("about.html")

@app.route('/efficient', methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]

    predictiondtree=modeldtree.predict_proba(final)
    predictionrfc=modelrfc.predict_proba(final)
    predictionlogreg=modellogreg.predict_proba(final)
    predictionsvm=modelsvm.predict_proba(final)
    
    outputdtree='{0:.{1}f}'.format(predictiondtree[0][1], 2)
    outputrfc='{0:.{1}f}'.format(predictionrfc[0][1], 2)
    outputlogreg='{0:.{1}f}'.format(predictionlogreg[0][1], 2)
    outputsvm='{0:.{1}f}'.format(predictionsvm[0][1], 2)

    return render_template('predict_output.html',
        dtree='{}'.format(outputdtree),
        rfc='{}'.format(outputrfc),
        logreg='{}'.format(outputlogreg),
        svm='{}'.format(outputsvm)
        )


@app.route('/evaluate', methods=['POST','GET'])
def performance():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]

    predictiondtree=modelpdtree.predict_proba(final)
    predictionrfc=modelprfc.predict_proba(final)
    predictionlogreg=modelplogreg.predict_proba(final)
    predictionsvm=modelpsvm.predict_proba(final)
    
    outputdtree='{0:.{1}f}'.format(predictiondtree[0][1], 2)
    outputrfc='{0:.{1}f}'.format(predictionrfc[0][1], 2)
    outputlogreg='{0:.{1}f}'.format(predictionlogreg[0][1], 2)
    outputsvm='{0:.{1}f}'.format(predictionsvm[0][1], 2)

    return render_template('evaluate_output.html',
        dtree='{}'.format(outputdtree),
        rfc='{}'.format(outputrfc),
        logreg='{}'.format(outputlogreg),
        svm='{}'.format(outputsvm)
        )
   

if __name__ == '__main__':
    app.run()
    
