import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from from_root import from_root


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join(from_root(), 'artifacts', 'preprocessor.pkl')
            model_path = os.path.join(from_root(), 'artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,LIMIT_BAL,SEX,EDUCATION, MARRIAGE, AGE, total_pay_amt, total_pay, total_bill_amt):
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.total_pay_amt = total_pay_amt
        self.total_pay = total_pay
        self.total_bill_amt = total_bill_amt

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL': [self.LIMIT_BAL],
                'SEX': [self.SEX],
                'EDUCATION': [self.EDUCATION],
                'MARRIAGE': [self.MARRIAGE],
                'AGE': [self.AGE],
                'total_pay_amt': [self.total_pay_amt],
                'total_pay': [self.total_pay],
                'total_bill_amt': [self.total_bill_amt]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)