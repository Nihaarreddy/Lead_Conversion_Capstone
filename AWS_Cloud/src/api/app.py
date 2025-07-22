# app.py

from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
from pyngrok import ngrok
from flask import send_file

app = Flask(__name__)
app.secret_key = "secure_key"

# Set MLflow URI and load model
mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:766957562594:mlflow-tracking-server/capstone-mlflow")  
# Same URI
model_uri = f"models:/LeadConversionPrediction/latest"
model = mlflow.pyfunc.load_model(model_uri)

INPUT_FIELDS = [
    "Lead Origin", "Lead Source", "Do Not Email", "Do Not Call",
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
    "Last Activity", "Country", "Specialization",
    "How did you hear about X Education", "What is your current occupation",
    "What matters most to you in choosing a course", "Search", "Magazine",
    "Newspaper Article", "X Education Forums", "Newspaper", "Digital Advertisement",
    "Through Recommendations", "Receive More Updates About Our Courses", "Tags",
    "Lead Quality", "Update me on Supply Chain Content", "Get updates on DM Content",
    "Lead Profile", "City", "Asymmetrique Activity Index", "Asymmetrique Profile Index",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score",
    "I agree to pay the amount through cheque", "A free copy of Mastering The Interview",
    "Last Notable Activity"
]

NUMERIC_FIELDS = [
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
    "Asymmetrique Activity Index", "Asymmetrique Profile Index",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score"
]

@app.route("/", methods=["GET", "POST"])
def home():
    values = {}
    if request.method == "POST":
        try:
            data = {}
            for field in INPUT_FIELDS:
                value = request.form.get(field)
                if field in NUMERIC_FIELDS:
                    try:
                        value = float(value)
                    except:
                        value = np.nan
                data[field] = value
            # Rename user input keys to match model training schema
            normalized_data = {k.lower().replace(" ", "_"): v for k, v in data.items()}
            input_df = pd.DataFrame([normalized_data])
            prediction = model.predict(input_df)[0]
            prediction_text = "Lead is Likely to Convert" if prediction == 1 else "Lead is Unlikely to Convert"
            flash(f"Prediction: {prediction_text} (Class={prediction})", "success")
            values = data  # Pre-fill form with submitted values
        except Exception as e:
            flash(f"❌ Error during prediction: {e}", "danger")
            values = request.form  # Could also pre-fill with submitted form

    return render_template("form.html", fields=INPUT_FIELDS, values=values)

@app.route("/drift_report")
def drift_report():
    file_path = os.path.join(os.path.dirname(__file__), 'static', 'drift_report.html')
    return send_file(file_path)

@app.route("/drift_train_vs_test")
def drift_train_vs_test():
    file_path = os.path.join(os.path.dirname(__file__), 'static', 'drift_train_vs_test.html')  # or actual file name
    return send_file(file_path)


if __name__ == "__main__":
    port = 8989
    public_url = ngrok.connect(port)
    print(f"🔗 Public URL: {public_url}")
    app.run(port=port)




