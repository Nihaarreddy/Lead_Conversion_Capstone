# Importing all the libraries required
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import pandas as pd
import numpy as np
import mlflow.pyfunc
from src.data_ingestion.db import write_table, read_table
import os
import re
# Set MLFlow tracking URI
mlflow.set_tracking_uri('file:///C:/Users/Minfy/Capstone_Project/mlruns')

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # For flash messages

# Load model from MLflow Model Registry
MLFLOW_MODEL_URI = "models:/Lead_Conversion_Prediction/1"
model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)

# Human-friendly input fields (used in form/UI)
INPUT_FIELDS = [
    "Lead Origin",
    "Lead Source",
    "Do Not Email",
    "Do Not Call",
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Last Activity",
    "Country",
    "Specialization",
    "How did you hear about X Education",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "Search",
    "Magazine",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "Receive More Updates About Our Courses",
    "Tags",
    "Lead Quality",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "Lead Profile",
    "City",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "I agree to pay the amount through cheque",
    "A free copy of Mastering The Interview",
    "Last Notable Activity"
]

# Numeric input fields (must be normalized to match column names)
NUMERIC_FIELDS = [
    "totalvisits",
    "total_time_spent_on_website",
    "page_views_per_visit",
    "asymmetrique_activity_index",
    "asymmetrique_profile_index",
    "asymmetrique_activity_score",
    "asymmetrique_profile_score"
]

# Final columns to match your PostgreSQL DB schema
FULL_COLUMNS = [
    "prospect_id", "lead_number", "lead_origin", "lead_source", "do_not_email", "do_not_call", "converted",
    "totalvisits", "total_time_spent_on_website", "page_views_per_visit", "last_activity", "country",
    "specialization", "how_did_you_hear_about_x_education", "what_is_your_current_occupation",
    "what_matters_most_to_you_in_choosing_a_course", "search", "magazine", "newspaper_article",
    "x_education_forums", "newspaper", "digital_advertisement", "through_recommendations",
    "receive_more_updates_about_our_courses", "tags", "lead_quality", "update_me_on_supply_chain_content",
    "get_updates_on_dm_content", "lead_profile", "city", "asymmetrique_activity_index",
    "asymmetrique_profile_index", "asymmetrique_activity_score", "asymmetrique_profile_score",
    "i_agree_to_pay_the_amount_through_cheque", "a_free_copy_of_mastering_the_interview", "last_notable_activity"
]


def normalize(col):
    return re.sub(r'[\s]+', '_', col.strip().lower())

NORMALIZED_INPUT_FIELDS = [normalize(col) for col in INPUT_FIELDS]


def normalize(col):
    return re.sub(r'[\s]+', '_', col.strip().lower())

def preprocess_single_form(form):
    """
    Convert user form into a normalized DataFrame matching DB column schema.
    """
    data = {}

    for field in INPUT_FIELDS:
        val = form.get(field)
        field_normalized = normalize(field)


        # Normalize and handle numerics
        if field_normalized in NUMERIC_FIELDS:
            try:
                val = float(val) if val not in [None, ''] else np.nan
            except:
                val = np.nan

        data[field_normalized] = [val]  # ✅ Use normalized field name as key

    # Add required DB fields not in form
    for col in ["prospect_id", "lead_number", "converted"]:
        data[col] = [None]

    df = pd.DataFrame(data)

    # ✅ Just to debug exactly what's included
    missing_cols = set(FULL_COLUMNS) - set(df.columns)
    if missing_cols:
        print("⚠️ Missing fields:", missing_cols)

    df = df[FULL_COLUMNS]  # ✅ reorder to match DB

    return df



def preprocess_batch_df(df):
    df.columns = [normalize(col) for col in df.columns]

    missing = set(FULL_COLUMNS) - set(df.columns)
    if missing:
        print(f"Missing fields in batch upload: {missing}")
        for col in missing:
            df[col] = None

    return df[FULL_COLUMNS]



@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Step 1: Get input
            input_df = preprocess_single_form(request.form)

            # Step 2: Predict
            model_input = input_df[NORMALIZED_INPUT_FIELDS]
            predicted_class = model.predict(model_input)[0]

            # Step 3: Save prediction
            input_df["converted"] = predicted_class  # Update converted
            write_table(input_df, table_name="user_batch_upload", if_exists="append")

            # Step 4: User feedback
            category = "Lead is likely to Convert" if predicted_class == 1 else "Lead is Unlikely to Convert"
            flash(f"Prediction: {category} (class={predicted_class})", "success")

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")

        return render_template("form.html", fields=INPUT_FIELDS, values=request.form)

    return render_template("form.html", fields=INPUT_FIELDS, values={})


@app.route("/batch-upload", methods=["GET", "POST"])
def batch_upload():
    if request.method == "POST":
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected", "danger")
            return redirect(request.url)

        try:
            df = pd.read_csv(file)
            df_processed = preprocess_batch_df(df)
            df_model_input = df_processed[NORMALIZED_INPUT_FIELDS]

            # Predict
            preds = model.predict(df_model_input)
            df_processed["Predicted Class"] = preds
            df_processed["Prediction"] = df_processed["Predicted Class"].map({
                1: "Lead is likely to Convert",
                0: "Lead is Unlikely to Convert"
            })

            # Save to database
            write_table(df_processed, "user_batch_upload")
            table_html = df_processed.to_html(classes='table table-striped', index=False)

            return render_template("batch_results.html", table_html=table_html, filename=file.filename)

        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")
            return redirect(request.url)

    return render_template("batch_upload.html")


@app.route('/drift-report')
def drift_report():
    report_path = os.path.join(os.getcwd(), "reports", "drift_full_vs_batch.html")
    if not os.path.exists(report_path):
        return "Drift report not found.", 404
    return send_file(report_path)


@app.route('/drift-train-vs-test')
def drift_train_vs_test():
    report_path = os.path.join(os.getcwd(), "reports", "drift_train_test.html")
    if not os.path.exists(report_path):
        return "Drift report not found.", 404
    return send_file(report_path)


if __name__ == "__main__":
    app.run(debug=True, port=8000)




