import numpy as np
import pickle
import os
from flask import Flask, request, render_template
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'input')
ALLOWED_EXTENSIONS = {'txt', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('breast_cancer_diagnosis_app.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('results.html', output=['No file part'])
    file = request.files['file']
    if file.filename == '':
        return render_template('results.html', output=['No selected file'])
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df_temp = pd.read_csv(filepath, sep='\t', header=None)
            df_temp.rename(columns={0: "sites", 1: "beta_value"}, inplace=True)
            df_temp.set_index('sites', inplace=True)
            df_temp = df_temp.T

            cpg_sites = [
                'cg00820405', 'cg06282596', 'cg06721601', 'cg12245706',
                'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
                'cg21113446', 'cg22300566'
            ]

            df_filtered_columns = df_temp[cpg_sites]
            print("\nFiltered DataFrame with specified CpG sites:")
            print(df_filtered_columns)

            output = pipeline_model(df_filtered_columns)
            return render_template('results.html', output=output)

        except Exception as e:
            print(f"Error processing file: {e}")
            return render_template('results.html', output=[f"Error processing file: {e}"])
    else:
        return render_template('results.html', output=['File type not allowed'])


def pipeline_model(x_test):
    output = []

    try:
        with open('tumor_stage_model.pkl', 'rb') as file:
            tumor_stage_model = pickle.load(file)
        with open("metastasis_status_model.pkl", 'rb') as file:
            metastasis_status_model = pickle.load(file)
        with open("subdivision_model.pkl", 'rb') as file:
            subdivision_model = pickle.load(file)
        with open("metastasis_site_model.pkl", 'rb') as file:
            metastasis_site_model = pickle.load(file)

        # Ensure x_test is a 2D array
        if len(x_test.shape) == 1:
            x_test = x_test.values.reshape(1, -1)

        # Log the shape of x_test for debugging
        print(f"x_test shape: {x_test.shape}")

        # Make predictions
        tumor_stage_predictions = tumor_stage_model.predict(x_test)
        print(f"Tumor Stage Predictions: {tumor_stage_predictions}")
        tumor_stage_predictions = int(tumor_stage_predictions[0])
        output.append(f'Tumor Stage: {tumor_stage_predictions}')

        if tumor_stage_predictions == 0:
            output.append('Patient is Tumor Free')
        else:
            metastasis_status_predictions = metastasis_status_model.predict(x_test)
            print(f"Metastasis Status Predictions: {metastasis_status_predictions}")
            metastasis_status_predictions = int(metastasis_status_predictions[0])
            output.append(f'Metastasis Status: {metastasis_status_predictions}')

            if metastasis_status_predictions == 0:
                output.append(f'M0: No metastasis')
            else:
                subdivision_predictions = subdivision_model.predict(x_test)
                print(f"Subdivision Predictions: {subdivision_predictions}")
                if subdivision_predictions.ndim > 1:
                    subdivision_predictions = subdivision_predictions[0]
                    print(f"Subdivision Predictions: {subdivision_predictions}")
                output.append(f'Anatomic Neoplasm Subdivision: {subdivision_predictions.tolist()}')

                metastasis_site_predictions = metastasis_site_model.predict(x_test)
                print(f"Metastasis Site Predictions: {metastasis_site_predictions}")
                if metastasis_site_predictions.ndim > 1:
                    metastasis_site_predictions = metastasis_site_predictions[0]
                    print(f"Metastasis Site Predictions: {metastasis_site_predictions}")
                output.append(f'Metastasis Site: {metastasis_site_predictions.tolist()}')

        return output
    except Exception as e:
        print(f"Error in pipeline_model: {e}")
        raise e


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import pickle
# import logging
#
# app = Flask(__name__)
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
#
# # Load the models
# try:
#     with open('subdivision_model.pkl', 'rb') as file:
#         subdivision_model = pickle.load(file)
#     with open('metastasis_site_model.pkl', 'rb') as file:
#         metastasis_site_model = pickle.load(file)
#     with open('metastasis_status_model.pkl', 'rb') as file:
#         metastasis_status_model = pickle.load(file)
#     with open('tumor_stage_model.pkl', 'rb') as file:
#         tumor_stage_model = pickle.load(file)
# except Exception as e:
#     logging.error(f"Error loading models: {str(e)}")
#
#
# def pipeline_model(df):
#     # Ensure DataFrame is in correct format
#     try:
#         logging.info(f"Shape of DataFrame passed to model: {df.shape}")
#         logging.info(f"DataFrame columns: {df.columns}")
#     except Exception as e:
#         logging.error(f"Error preparing DataFrame for model: {str(e)}")
#         return {"error": str(e)}
#
#     # Make predictions
#     try:
#         logging.info("Predicting tumor stage")
#         tumor_stage_predictions = tumor_stage_model.predict(df)
#         results = {'Tumor Stage': tumor_stage_predictions[0]}
#         if tumor_stage_predictions[0] == 0:
#             results['Diagnosis'] = 'Patient is Tumor Free'
#         else:
#             logging.info("Predicting metastasis status")
#             metastasis_status_predictions = metastasis_status_model.predict(df)
#             results['Metastasis Status'] = (metastasis_status_predictions[0])
#             if metastasis_status_predictions[0] == 0:
#                 results['Metastasis'] = 'M0: No metastasis'
#             else:
#                 logging.info("Predicting anatomic neoplasm subdivision")
#                 subdivision_predictions = subdivision_model.predict(df)
#                 results['Anatomic Neoplasm Subdivision'] = (subdivision_predictions[0])
#                 logging.info("Predicting metastasis site")
#                 metastasis_site_predictions = metastasis_site_model.predict(df)
#                 results['Metastasis Site'] = (metastasis_site_predictions[0])
#     except Exception as e:
#         logging.error(f"Error making predictions: {str(e)}")
#         return {"error": str(e)}
#
#     return results
#
#
# @app.route('/')
# def index():
#     return render_template('breast_cancer_diagnosis_app.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file part in the request"})
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No file selected"})
#
#     try:
#         if file and (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
#             df = pd.read_csv(file, index_col=0, sep='\t' if file.filename.endswith('.txt') else ',')
#             df = df.T
#             df = df.dropna(axis=1, how='any')
#             df_features = df[
#                 ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900',
#                  'cg20253855', 'cg21113446', 'cg22300566']]
#             logging.info(f"Data passed to model: {df_features}")
#         else:
#             return jsonify({"success": False, "error": "Unsupported file format"})
#     except Exception as e:
#         logging.error(f"Error processing file: {str(e)}")
#         return jsonify({"success": False, "error": str(e)})
#
#     try:
#         results = pipeline_model(df_features)
#         return jsonify({"success": True, "results": results})  # Return JSON response with results
#     except Exception as e:
#         logging.error(f"Error running pipeline model: {str(e)}")
#         return jsonify({"success": False, "error": str(e)})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
