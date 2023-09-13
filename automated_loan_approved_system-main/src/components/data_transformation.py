import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder,LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    outputprocessor_obj_file_path=os.path.join('artifacts','outputprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            logging.info('Pipeline Initiated')

            num_columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                                    'Loan_Amount_Term', 'Credit_History']
            cat_columns=['Gender', 'Married', 'Dependents',
                                                    'Education', 'Self_Employed', 'Property_Area']



            ## Numerical Pipeline
            num_pipe=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )

            # Categorigal Pipeline
            cat_pipe=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('transformer',OneHotEncoder(categories='auto'))
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipe,num_columns),
            ('cat_pipeline',cat_pipe,cat_columns)
            ])

            return preprocessor
        
            logging.info('Pipeline Completed')



        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
    
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Loan_Status'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            input_feature_train_df = input_feature_train_df.drop('Loan_ID',axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            input_feature_test_df=input_feature_test_df.drop('Loan_ID',axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            preprocessing_obj.fit(input_feature_train_df)
            input_feature_train_arr=preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            #Performing LabelEncoding on the Target Column
            label_encoder=LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            target_feature_train_df=label_encoder.transform(target_feature_train_df)
            target_feature_test_df=label_encoder.transform(target_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #Saving the preprocessor.pkl file
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            #Saving the encoder.pkl file

            save_object(
                file_path=self.data_transformation_config.outputprocessor_obj_file_path,
                obj=label_encoder
            )

            logging.info('OutputProcessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)