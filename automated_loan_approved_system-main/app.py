from src.logger import logging
from src.exception import CustomException
from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import datetime as dt
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.utils import load_object
import os
import sys

app=Flask(__name__)

@app.route('/')
def show(): 
    return render_template('index.html')

@app.route("/review",methods=['GET','POST'])
def find():
    if request.method=="POST":

        gender1=request.form['gender']
        # print(age1)
        married1=request.form.get('married')
        dependents1=request.form.get('dependents')
        education1=request.form.get('education')
        selfemployed1=request.form.get('selfemployed')
        appincome1=int(request.form.get('appincome'))
        coappincome1=float(request.form.get('coappincome'))
        lamount1=float(request.form.get('lamount'))
        ltamount1=float(request.form.get('ltamount'))
        credith1=float(request.form.get('credith'))
        property1=request.form.get('property')


        data=CustomData(Gender=gender1,
                        Married=married1,
                        Dependents=dependents1,
                        Education=education1,
                        Self_Employed=selfemployed1,
                        ApplicantIncome=appincome1,
                        CoapplicantIncome=coappincome1,
                        LoanAmount=lamount1,
                        Loan_Amount_Term=ltamount1,
                        Credit_History=credith1,
                        Property_Area=property1)
        
        final_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_data)
        pred=pred.reshape(-1)
        pred=pred.astype('int')
        
        #trasforming the output
        output_processor = os.path.join('artifacts','outputprocessor.pkl')

        processor=load_object(output_processor)

        res = processor.inverse_transform(pred)[0]
        
        if res=='Y':
            con = 'Yes the person is Eligible for Loan approval'
        else:
            con = 'No the person is Not Eligible for Loan Approval'

        

        return render_template('results.html',final_result=con)
        


if __name__ == '__main__':
    app.run(debug=True)