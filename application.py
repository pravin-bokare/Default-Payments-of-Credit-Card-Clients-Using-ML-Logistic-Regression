from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX=float(request.form.get('SEX')),
            EDUCATION=float(request.form.get('EDUCATION')),
            MARRIAGE=float(request.form.get('MARRIAGE')),
            AGE=float(request.form.get('AGE')),
            total_pay_amt=float(request.form.get('total_pay_amt')),
            total_pay=request.form.get('total_pay'),
            total_bill_amt=request.form.get('total_bill_amt'),
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)