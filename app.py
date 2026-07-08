from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    data = CustomData(
        age=int(request.form.get('Age')),
        sex=request.form.get('Sex'),
        chest_pain_type=request.form.get('ChestPainType'),
        resting_bp=int(request.form.get('RestingBP')),
        cholesterol=int(request.form.get('Cholesterol')),
        fasting_bs=int(request.form.get('FastingBS')),
        max_hr=int(request.form.get('MaxHR')),
        resting_ecg=request.form.get('RestingECG'),
        exercise_angina=request.form.get('ExerciseAngina'),
        oldpeak=float(request.form.get('Oldpeak')),
        st_slope=request.form.get('ST_Slope'),
    )

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)[0]

    prediction_text = (
        "Heart Disease Risk Detected" if result == 1
        else "No Heart Disease Risk Detected"
    )

    return render_template('home.html', prediction=result, prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
