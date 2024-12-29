# # Imports pandas library and loads our data into a variable called dataset
#
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
#
# def read_csv():
#     file_path = '/Users/thaarun/PythonProjects/AIResearch/output/results/MLDatasetFeatures.csv'
#     df = pd.read_csv(file_path, sep='\t')
#     return df
#
#
# # print(df.head())
#
# def smote(a, b):
#     model = SMOTE(random_state=8)
#     X, y = model.fit_resample(a, b)
#     return X, y
#
#
# def stage(x):
#     if x == 'Stage I':
#         stage_val = 1
#     elif x == 'Stage II':
#         stage_val = 2
#     elif x == 'Stage IIB':
#         stage_val = 2
#     elif 'Stage III' in x:
#         stage_val = 3
#     elif x == 'Stage IV':
#         stage_val = 4
#     else:
#         stage_val = 0
#     return stage_val
#
#
# def tumor_stage_prediction():
#     df_temp = read_csv()
#     df_X = df_temp.drop(columns=['patient_file_name', 'patient_name', 'metastatic_status',
#                                  'anatomic_neoplasm_subdivision', 'metastasis_site'])
#     print(df_X.columns[0:4])
#
#     df_t_tumor_stage = df_temp[['tumor_stage']]
#     vectorized_stage = np.vectorize(stage)
#     df_t_tumor_stage['stage'] = vectorized_stage(df_t_tumor_stage['tumor_stage'])
#     df_y_tumor_stage = df_t_tumor_stage.drop(columns=['tumor_stage'])
#     print(df_y_tumor_stage)
#     X_train_smoted, y_train_smoted = smote(df_X, df_y_tumor_stage)
#
#     print(X_train_smoted)
#     print(y_train_smoted)
#
#
# def metastasis_status(x):
#     if x == 'M0':
#         status_val = 0
#     elif x == 'M1':
#         status_val = 1
#     else:
#         status_val = 2
#     return status_val
#
#
# def metastasis_prediction():
#     df_temp = read_csv()
#     df_X = df_temp.drop(columns=['patient_file_name', 'patient_name', 'anatomic_neoplasm_subdivision',
#                                  'metastasis_site'])
#     df_t_metastatic_status = df_temp[['metastatic_status']]
#     vectorized_status = np.vectorize(metastasis_status)
#     df_t_metastatic_status['metastatic_status'] = vectorized_status(df_t_metastatic_status['metastasis_status'])
#     df_y_metastatic_status = df_t_metastatic_status.drop(columns=['metastasis_status'])
#     print(df_y_metastatic_status)
#     X_train_smoted, y_train_smoted = smote(df_X, df_y_metastatic_status)
#
#     print(X_train_smoted)
#     print(y_train_smoted)
#
#
# def subdivision(x):
#     if x == 'Left Lower':
#         subdivision_val = 0
#     elif x == 'Left Upper':
#         subdivision_val = 1
#     elif x == 'Right Lower':
#         subdivision_val = 2
#     elif x == 'Right Upper':
#         subdivision_val = 3
#     return subdivision_val
#
#
# def anatomic_subdivision_prediction():
#     df_temp = read_csv()
#     df_X = df_temp.drop(columns=['patient_file_name', 'patient_name', 'metastatic_status', 'metastasis_site'])
#     df_t_subdivision = df_temp[['anatomic_neoplasm_subdivision']]
#     vectorized_status = np.vectorize(subdivision)
#     df_t_subdivision['anatomic_neoplasm_subdivision'] = vectorized_status(df_t_subdivision['anatomic_neoplasm_subdivision'])
#     df_y_subdivision = df_t_subdivision.drop(columns=['tumor_stage'])
#     print(df_y_subdivision)
#     X_train_smoted, y_train_smoted = smote(df_X, df_y_subdivision)
#
#     print(X_train_smoted)
#     print(y_train_smoted)
#
#
# def site(x):
#     if x == 'Left Lower':
#         subdivision_val = 0
#     elif x == 'Left Upper':
#         subdivision_val = 1
#     elif x == 'Right Lower':
#         subdivision_val = 2
#     elif x == 'Right Upper':
#         subdivision_val = 3
#     return subdivision_val
#
#
# def metastasis_site_prediction():
#
#
# if __name__ == "__main__":
#     tumor_stage_prediction()
#     metastasis_prediction()
#     anatomic_subdivision_prediction()
#
#
