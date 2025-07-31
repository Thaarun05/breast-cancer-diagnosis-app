import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn import model_selection
import pickle
# import tensorflow_decision_forests as tfdf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets, linear_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils import to_categorical


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def read_csv():
    file_path = '/Users/thaarun/PythonProjects/AIResearch/output/results/MLDatasetFeatures.txt'
    df = pd.read_csv(file_path, sep='\t')
    return df


def smote(x, y):
    model = SMOTE(random_state=8, k_neighbors=4)
    X_resampled, y_resampled = model.fit_resample(x, y)
    return X_resampled, y_resampled


def evaluation(y_true, y_pred):
    # Print Accuracy, Recall, F1 Score, and Precision metrics.
    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    precision = metrics.precision_score(y_true, y_pred, average='weighted')

    print('Evaluation Metrics:')
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('Precision:', precision)

    # Print Confusion Matrix
    # print('\nConfusion Matrix:')
    # print(confusion_matrix(y_true, y_pred))


def stage(x):
    if x == 'Stage I':
        stage_val = 1
    elif x == 'Stage II':
        stage_val = 2
    elif x == 'Stage IIB':
        stage_val = 2
    elif 'Stage III' in x:
        stage_val = 3
    elif x == 'Stage IV':
        stage_val = 4
    else:
        stage_val = 0
    return stage_val


def tumor_stage_prediction():
    df_temp = pd.read_csv('/Users/thaarun/PythonProjects/AIResearch/output/results/MLDatasetFeatures.txt',
                          delimiter='\t')

    # print(df_temp)

    # df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage', 'metastatic_status',
    #                              'anatomic_neoplasm_subdivision', 'metastasis_site'])

    df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage',
                                 'anatomic_neoplasm_subdivision', 'metastasis_site'])

    df_X = df_x.dropna(axis=1, how='any')

    # threshold = len(df_x) * 0.8  # Keep columns with at least 80% non-null values
    #
    # # Drop columns with less than the specified threshold non-null values
    # df_X = df_x.dropna(axis=1, thresh=threshold)
    #
    # has_nan_values = df_X.isna().any().any()
    # if has_nan_values:
    #     print("DataFrame has NaN values.")
    # else:
    #     print("DataFrame does not have any NaN values.")

    # print(df_X)
    # df_X = df_X.drop([0, 3, 14])

    # df_X = df_X.reset_index(drop=True)

    duplicated_sites = df_X.loc[[0, 3, 14]]
    duplicated_sites_X = duplicated_sites[
        ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
         'cg21113446', 'cg22300566']]

    df_top_ten_sites = df_X[
        ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
         'cg21113446', 'cg22300566']]
    # print('df_top_ten_sites:')
    print('Before smote')
    print(df_top_ten_sites)
    df_top_ten_sites = df_top_ten_sites.drop([0, 3, 14])
    # print('top 10 sites with indexes 0,3,14 dropped')
    # print(df_top_ten_sites)
    # print('duplicated sites:')
    # print(duplicated_sites_X)

    df_t_tumor_stage = df_temp[['tumor_stage']]
    df_t_tumor_stage = df_temp[['tumor_stage', 'metastatic_status']]

    # MX_indexes = df_t_tumor_stage[df_t_tumor_stage['metastatic_status'] == 'MX'].index

    # df_MX_patients = df_t_tumor_stage[df_t_tumor_stage['metastatic_status'] == 'MX']

    vectorized_stage = np.vectorize(stage)
    df_t_tumor_stage['stage'] = vectorized_stage(df_t_tumor_stage['tumor_stage'])
    # print(df_t_tumor_stage)
    df_y_tumor_stage = df_t_tumor_stage.drop(columns=['tumor_stage', 'metastatic_status'])
    print(df_y_tumor_stage)
    df_y_tumor_stage_before = df_y_tumor_stage

    duplicated_patients_y = df_y_tumor_stage.loc[[0, 3, 14]]
    # print('Duplicated tumor stage:')
    # print(duplicated_patients_y)
    df_y_tumor_stage = df_y_tumor_stage.drop([0, 3, 14])
    df_y_tumor_stage = df_y_tumor_stage.reset_index(drop=True)
    # print(df_y_tumor_stage)

    # print(duplicated_sites_X)

    # df_y_tumor_stage = pd.concat([df_y_tumor_stage, duplicated_patients_y], ignore_index=True)
    # print(df_y_tumor_stage)
    # print(df_top_ten_sites)
    # print("Df size")

    # print(df_top_ten_sites.shape)
    # print(df_y_tumor_stage.shape)

    x_smoted, y_smoted = smote(df_top_ten_sites, df_y_tumor_stage)

    # print(x_smoted)
    # print(y_smoted)

    # print("Smoted size")
    # print(x_smoted.shape)
    # print(y_smoted.shape)

    X_sites = pd.concat([x_smoted, duplicated_sites_X], ignore_index=True)
    Y_stages = pd.concat([y_smoted, duplicated_patients_y], ignore_index=True)
    print('After Smote')
    print(X_sites)
    print(Y_stages)

    stages_before = df_y_tumor_stage_before['stage'].value_counts().sort_index()
    stages_after = Y_stages['stage'].value_counts().sort_index()

    # Create bar plots
    stages = stages_before.index.union(stages_after.index)
    plt.figure(figsize=(14, 7))  # Increase figure size for better visibility

    plt.subplot(1, 2, 1)
    plt.bar(stages_before.index, stages_before.values, color='blue', alpha=0.7, label='Before SMOTE')
    plt.xticks(stages, stages, fontsize=12)  # Increase x-tick labels font size
    plt.xlabel('Tumor Stage', fontsize=14)  # Increase x-axis label font size
    plt.ylabel('Number of Samples', fontsize=14)  # Increase y-axis label font size
    plt.title('Tumor Stage Distribution Before SMOTE', fontsize=16)  # Increase title font size
    plt.legend(fontsize=12)  # Increase legend font size

    plt.subplot(1, 2, 2)
    plt.bar(stages_after.index, stages_after.values, color='orange', alpha=0.7, label='After SMOTE')
    plt.xticks(stages, stages, fontsize=12)  # Increase x-tick labels font size
    plt.xlabel('Tumor Stage', fontsize=14)  # Increase x-axis label font size
    plt.ylabel('Number of Samples', fontsize=14)  # Increase y-axis label font size
    plt.title('Tumor Stage Distribution After SMOTE', fontsize=16)  # Increase title font size
    plt.legend(fontsize=12)  # Increase legend font size

    plt.tight_layout()
    plt.show()

    # df_ML_dataset = X_sites.join(Y_stages)
    # print(df_ML_dataset)
    # df_ML_dataset.to_csv('/Users/thaarun/PythonProjects/AIResearch/output/results/tumor_stage_pred_data_2.csv',
    #                      index=False,
    #                      sep=',')

    # print(X_sites)
    # print(Y_stages)
    #
    # print("MLDataset size")
    # print(X_sites.shape)
    # print(Y_stages.shape)

    # X_train, X_test, y_train, y_test = train_test_split(X_sites, Y_stages, test_size=0.25, random_state=11)

    # print('X_train:\n', X_train)
    # print('X_train shape:', X_train.shape)
    #
    # print('X_test:\n', X_test)
    # print('X_test shape:', X_test.shape)
    #
    # print('y_train:\n', y_train)
    # print('y_train shape:', y_train.shape)
    #
    # print('y_test:\n', y_test)
    # print('y_test shape:', y_test.shape)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # print(X_train_scaled)
    # print(X_test_scaled)

    # Get unique class labels from y_train and y_test
    # unique_labels_train = np.unique(y_train)
    # unique_labels_test = np.unique(y_test)
    #
    # Ensure that classes parameter includes all valid labels from both training and testing sets
    # all_classes = np.unique(np.concatenate([unique_labels_train, unique_labels_test]))
    # # Compute class weights
    # class_weights = compute_class_weight('balanced', classes=all_classes, y=y_train)
    #
    # # Create a dictionary of class weights
    # class_weight_dict = dict(zip(all_classes, class_weights))

    # Support Vector Machine Classifier
    # svm = SVC(C=100, class_weight='balanced', kernel='linear')
    # # svm = SVC(C=100, class_weight='balanced')
    # # Fitting Model to the train set
    # svm.fit(X_train, y_train)
    #
    # # Predicting on the test set
    # y_pred = svm.predict(X_test)

    # with open('tumor_stage_model.pkl', 'wb') as file:
    #     pickle.dump(svm, file)
    #
    # return None

    # Multilayer neural network
    # model_1 = Sequential()
    # model_1.add(Dense(64, activation='relu', input_shape=(10,)))
    # model_1.add(Dense(32, activation='relu'))
    # model_1.add(Dense(16, activation='relu'))
    # model_1.add(Dense(1, activation='relu'))
    # model_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model_1.summary()
    # model_1.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)
    # y_pred = model_1.predict(X_test)

    # MLPClassifer
    # param_grid = {
    #     'hidden_layer_sizes': [(50, 25), (100, 50), (150, 100)],
    #     'max_iter': [100, 200, 300],
    #     'alpha': [0.0001, 0.001, 0.01],
    # }
    # # batch_size = [10, 20, 40, 60, 80, 100]
    # # epochs = [10, 50, 100]
    # # param_grid = dict(batch_size=batch_size, epochs=epochs)
    #
    # mlp_model = MLPClassifier(random_state=11, early_stopping=True)
    # grid_search = GridSearchCV(mlp_model, param_grid, cv=5, scoring='accuracy')
    # # grid_search = GridSearchCV(model_1, param_grid, cv=5, scoring='accuracy')
    #
    # # grid_search = RandomizedSearchCV(mlp_model, param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    # grid_search.fit(X_train, y_train)
    #
    # best_params = grid_search.best_params_
    # print("Best Parameters:", best_params)
    #
    # best_model = grid_search.best_estimator_
    #
    # accuracy = best_model.score(X_test, y_test)
    # print("Accuracy:", accuracy)

    # mlp_model = MLPClassifier(hidden_layer_sizes=(150, 100), alpha=0.0001, max_iter=100, random_state=11, early_stopping=True)

    # Found by Grid Search mlp_model = MLPClassifier(hidden_layer_sizes=(200, 125), max_iter=100,
    # validation_fraction=0.1, random_state=11, early_stopping=True)

    # Found by Random Search
    # mlp_model = MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=300, validation_fraction=0.1, random_state=11, early_stopping=True)
    # mlp_model.fit(X_train, y_train)
    # y_pred = mlp_model.predict(X_test)

    # Random forest classifer
    # clf = RandomForestClassifier(n_estimators = 50)
    # # multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    #
    # print("Actual values predicted")
    # print(y_pred)
    # print("Y test values")
    # print(y_test)
    # print(type(y_test))
    # print(type(y_pred))
    #
    # # Evaluating model
    # evaluation(y_test, y_pred)

    # Plot loss over epochs
    # plt.figure(figsize=(10, 5))
    # plt.plot(mlp_model.loss_curve_)
    # plt.title('MLP Loss Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()
    #
    # # Plot accuracy over epochs
    # plt.figure(figsize=(10, 5))
    # plt.plot(mlp_model.validation_scores_)
    # plt.title('MLP Accuracy Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.show()


def metastasis_status(x):
    if x == 'M0':
        status_val = 0
    elif x == 'M1':
        status_val = 1
    return status_val


def metastasis_prediction():
    df_temp = read_csv()
    df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage', 'anatomic_neoplasm_subdivision',
                                 'metastasis_site'])
    df_X = df_x.dropna(axis=1, how='any')
    # print(df_X)

    df_t_metastatic_status = df_temp[['metastatic_status']]
    print(df_t_metastatic_status)

    df_MX_patients = df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX']
    # print(df_MX_patients)

    # print(df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX'].index.values)
    MX_indexes = df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX'].index.values
    # print(MX_indexes)

    df_t_metastatic_status.drop(df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX'].index,
                                inplace=True)
    # print(df_t_metastatic_status)
    df_t_metastatic_status.reset_index(drop=True, inplace=True)
    # print(df_t_metastatic_status)

    df_top_ten_sites = df_X[
        ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
         'cg21113446', 'cg22300566']]
    # print('This is df_top_ten_sites')
    # print(df_top_ten_sites)
    # df_MX_sites = df_top_ten_sites[MX_indexes]
    # print(df_MX_sites)
    # print(df_MX_sites.shape)

    df_MX_sites = df_top_ten_sites.iloc[MX_indexes]
    df_MX_sites.reset_index(drop=True, inplace=True)
    # print(df_MX_sites)

    df_top_ten_sites_train = df_top_ten_sites.drop(MX_indexes)

    # print('This is df_top_ten_sites')
    # print(df_top_ten_sites_train)
    # print(df_top_ten_sites_train.shape)

    vectorized_status = np.vectorize(metastasis_status)
    df_t_metastatic_status['metastatic_status'] = vectorized_status(df_t_metastatic_status['metastatic_status'])
    # df_MX_patients['metastatic_status'] = vectorized_status(df_MX_patients['metastatic_status'])
    # print(df_t_metastatic_status)
    # print(df_t_metastatic_status.shape)
    y_train = df_t_metastatic_status
    # print(y_train)
    # print(y_train.shape)

    X_train_smoted, y_train_smoted = smote(df_top_ten_sites_train, y_train)
    # print(X_train_smoted)
    # print(y_train_smoted)

    status_before = df_t_metastatic_status['metastatic_status'].value_counts().sort_index()
    status_after = y_train_smoted['metastatic_status'].value_counts().sort_index()
    #
    # Create bar plots
    # Create bar plots with enhanced readability
    statuses = sorted(set(status_before.index).union(status_after.index))
    plt.figure(figsize=(14, 7))  # Increase figure size for better visibility

    plt.subplot(1, 2, 1)
    plt.bar(status_before.index, status_before.values, color='blue', alpha=0.7, label='Before SMOTE')
    plt.xticks(statuses, statuses, fontsize=12)  # Increase x-tick labels font size
    plt.xlabel('Metastasis Status', fontsize=14)  # Increase x-axis label font size
    plt.ylabel('Number of Samples', fontsize=14)  # Increase y-axis label font size
    plt.title('Metastasis Status Distribution Before SMOTE', fontsize=16)  # Increase title font size
    plt.legend(fontsize=12)  # Increase legend font size

    plt.subplot(1, 2, 2)
    plt.bar(status_after.index, status_after.values, color='orange', alpha=0.7, label='After SMOTE')
    plt.xticks(statuses, statuses, fontsize=12)  # Increase x-tick labels font size
    plt.xlabel('Metastasis Status', fontsize=14)  # Increase x-axis label font size
    plt.ylabel('Number of Samples', fontsize=14)  # Increase y-axis label font size
    plt.title('Metastasis Status Distribution After SMOTE', fontsize=16)  # Increase title font size
    plt.legend(fontsize=12)  # Increase legend font size

    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_train_smoted, y_train_smoted, test_size=0.25, random_state=11)

    # print(y_test)

    # X_test = pd.concat([X_test, df_MX_sites], ignore_index=True)
    # y_test = pd.concat([y_test, df_MX_patients], ignore_index=True)
    #
    # y_test = y_test.replace({'MX': -1, '1': 1})

    # print(X_test)
    # print(y_test)

    #
    # print(y_train.shape)
    # print(y_test.shape)

    # print("X_train contents:\n", X_train)
    # print("X_train type:", type(X_train))
    #
    # print("\nX_test contents:\n", X_test)
    # print("X_test type:", type(X_test))
    #
    # print("\ny_train contents:\n", y_train)
    # print("y_train type:", type(y_train))
    #
    # print("\ny_test contents:\n", y_test)
    # print("y_test type:", type(y_test))

    # train_df = pd.concat([X_train, y_train], axis=1)
    # test_df = pd.concat([X_test, y_test], axis=1)

    # print(train_df)
    # print(test_df)

    # Support Vector Machine Classifier
    # svm = SVC(C=100, class_weight='balanced', kernel='linear')
    svm = SVC(C=100, class_weight='balanced')
    # Fitting Model to the train set
    svm.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = svm.predict(X_test)
    MX_pred = svm.predict(df_MX_sites)

    # with open('metastasis_status_model.pkl', 'wb') as file:
    #     pickle.dump(svm, file)
    #
    # return None

    # reg = linear_model.LogisticRegression()
    # reg.fit(X_train, y_train)
    # y_pred = reg.predict(X_test)
    #
    print("Actual values predicted")
    print(y_pred)
    print("Y test values")
    print(y_test)
    print(type(y_test))
    print(type(y_pred))
    print("MX Patients predictions: ")
    print(MX_pred)

    # print("Weights:")
    # print("Coefficients: \n", reg.coef_)

    # TF-DF
    # train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
    # test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")
    # model = tfdf.keras.RandomForestModel()
    # model.fit(train_ds)
    # model.summary()
    # model.compile(metrics=["accuracy"])
    # model.evaluate(test_ds, return_dict=True)

    # Scikit RF model
    # rf = RandomForestClassifier()
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    #
    # print("Actual values predicted")
    # print(y_pred)
    # print("Y test values")
    # print(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    #
    # support_vectors = svm.support_vectors_
    # dual_coefficients = svm.dual_coef_
    #
    # intercept = svm.intercept_
    #
    # print("Intercept:", intercept)
    #
    # print("Number of support vectors:", len(support_vectors))
    # print("Dual coefficients:", dual_coefficients)
    # print("Intercept:", intercept)

    # Evaluating model
    # evaluation(y_test, y_pred)


def subdivision(x):
    if 'Left Lower' in x:
        subdivision_val = 0
    elif 'Left Upper' in x:
        subdivision_val = 1
    elif 'Right Lower' in x:
        subdivision_val = 2
    elif 'Right Upper' in x:
        subdivision_val = 3
    else:
        subdivision_val = 4
    return subdivision_val


def preprocess_and_encode(df, target_column, encoding_func):
    df_X = df.drop(columns=['patient_file_name', 'patient_name', *target_column])
    df_t = df[target_column].copy()
    df_t[target_column[0]] = df_t[target_column[0]].apply(encoding_func)  # Apply encoding function to each value
    df_y = df_t
    return df_X, df_y


def encoding(x):
    return [site.strip() for site in x.split('|')]


def anatomic_subdivision_prediction():
    df_temp = read_csv()
    df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage',
                                 'metastasis_site'])
    df_X = df_x.dropna(axis=1, how='any')
    # print(df_X)

    df_t_subdivision = df_temp[['anatomic_neoplasm_subdivision']]
    # print(df_t_subdivision)

    df_X, df_y = preprocess_and_encode(df_temp, ['anatomic_neoplasm_subdivision'], encoding)
    #
    mlb = MultiLabelBinarizer()
    y_encoded = pd.DataFrame(mlb.fit_transform(df_t_subdivision['anatomic_neoplasm_subdivision'].str.split('|')),
                             columns=mlb.classes_)
    # print(y_encoded)

    # y_encoded.to_csv('/Users/thaarun/PythonProjects/AIResearch/output/results/subdivision_encoded.csv',
    # index=False, sep=',')

    df_top_ten_sites = df_X[
        ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
         'cg21113446', 'cg22300566']]
    print('This is df_top_ten_sites')
    print(df_top_ten_sites)
    # df_MX_sites = df_top_ten_sites[MX_indexes]
    # print(df_MX_sites)
    # print(df_MX_sites.shape)

    # df_MX_sites = df_top_ten_sites.iloc[MX_indexes]
    # df_MX_sites.reset_index(drop=True, inplace=True)
    # print(df_MX_sites)

    # df_top_ten_sites_train = df_top_ten_sites.drop(MX_indexes)

    # print('This is df_top_ten_sites')
    # print(df_top_ten_sites_train)
    # print(df_top_ten_sites_train.shape)

    # vectorized_status = np.vectorize(subdivision)
    # df_t_subdivision['subdivision'] = vectorized_status(df_t_subdivision['anatomic_neoplasm_subdivision'])
    # print(df_t_subdivision)
    # print(df_t_metastatic_status.shape)
    # y_train = df_t_metastatic_status
    # print(y_train)
    # print(y_train.shape)
    #
    # X_train_smoted, y_train_smoted = smote(df_top_ten_sites, y_encoded)
    # print(X_train_smoted)
    # print(y_train_smoted)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_train_smoted, y_train_smoted, test_size=0.25,
    # random_state=11)

    # subdivision_counts = y_encoded.sum(axis=0)
    #
    # # Plot the distribution
    # plt.figure(figsize=(10, 6))
    # plt.bar(subdivision_counts.index, subdivision_counts.values, color='skyblue', alpha=0.7)
    # plt.xlabel('Anatomic Subdivision')
    # plt.ylabel('Number of Samples')
    # plt.title('Anatomic Subdivision Distribution')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(df_top_ten_sites, y_encoded, test_size=0.25, random_state=11)

    print(y_test)
    #
    # X_test = pd.concat([X_test, df_MX_sites], ignore_index=True)
    # y_test = pd.concat([y_test, df_MX_patients], ignore_index=True)
    # #
    # # # print(X_test)
    # # # print(y_test)
    # #
    # # # print(y_train.shape)
    # # # print(y_test.shape)
    # #
    # forest = RandomForestClassifier(random_state=1)
    # multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
    # multi_target_forest.fit(X_train, y_train).predict(X_test)
    # y_pred = multi_target_forest.predict(X_test)

    # with open("subdivision_model.pkl", 'wb') as file:
    #     pickle.dump(multi_target_forest, file)
    #
    # return None

    # return multi_target_forest
    #
    # # # Support Vector Machine Classifier
    # # svm = SVC(C=100, class_weight='balanced')
    # #
    # # # Fitting Model to the train set
    # # svm.fit(X_train, y_train)
    # #
    # Predicting on the test set
    # print("Actual values predicted")
    # print(y_pred)
    # print("Y test values")
    # print(y_test)
    # print(type(y_test))
    # print(type(y_pred))
    #
    # # Evaluating model
    # evaluation(y_test, y_pred)


def metastasis_site_prediction ():
    df_temp = read_csv()
    df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage', 'anatomic_neoplasm_subdivision'])
    df_X = df_x.dropna(axis=1, how='any')
    # print(df_X)
    df_top_ten_sites = df_X[
        ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
         'cg21113446', 'cg22300566']]

    # print(df_top_ten_sites)
    # print(df_top_ten_sites.shape)
    # df_t_sites = df_temp[['metastasis_site', 'metastatic_status']]

    df_t_sites = df_temp[['metastasis_site']]
    # print(df_t_sites)

    df_X, df_y = preprocess_and_encode(df_temp, ['metastasis_site'], encoding)
    #
    mlb = MultiLabelBinarizer()
    y_encoded = pd.DataFrame(mlb.fit_transform(df_t_sites['metastasis_site'].str.split('|')), columns=mlb.classes_)
    # print(y_encoded)
    #
    # # print(y_encoded.shape)
    y_encoded.to_csv('/Users/thaarun/PythonProjects/AIResearch/output/results/metastasis_site_encoded.csv', index=False, sep=',')
    #
    # # X_train_smoted, y_train_smoted = smote(df_X, y_encoded)
    # # print(X_train_smoted)
    # # print(y_train_smoted)
    #
    X_train, X_test, y_train, y_test = train_test_split(df_top_ten_sites, y_encoded, test_size=0.25, random_state=11)
    # # print(X_train)
    # # print(X_test)
    # # print(y_train)
    # # print(y_test)
    # #
    # # print(X_train.shape)
    # # print(y_train.shape)
    #
    # tree = DecisionTreeClassifier(random_state=1, max_depth=2)
    # # neigh = KNeighborsClassifier(n_neighbors=3)
    # multi_target_forest = MultiOutputClassifier(tree, n_jobs=2)
    # # multi_target_forest = MultiOutputClassifier(neigh, n_jobs=2)
    # multi_target_forest.fit(X_train, y_train).predict(X_test)

    # metastasis_site_model = "metastasis_site_model.pkl"
    # with open("metastasis_site_model.pkl", 'wb') as file:
    #     pickle.dump(multi_target_forest, file)
    #
    # return None

    # # Support Vector Machine Classifier
    # # svm = SVC(C=100, class_weight='balanced')
    # #
    # # Fitting Model to the train set
    # # svm.fit(X_train, y_train)
    # #
    # Predicting on the test set`

    # y_pred = multi_target_forest.predict(X_test)
    # print("Actual values predicted")
    # print(y_pred)
    # print("Y test values")
    # print(y_test)
    # # print(type(y_test))
    # # print(type(y_pred))
    #
    # # # Evaluating model
    # evaluation(y_test, y_pred)


def pipeline_model(x_test):
    # Load the models
    # tumor_stage_model = joblib.load('tumor_stage_model.pkl')
    # metastasis_model = joblib.load('metastasis_model.pkl')
    # anatomic_subdivision_model = joblib.load('anatomic_subdivision_model.pkl')
    with open('tumor_stage_model.pkl', 'rb') as file:
        tumor_stage_model = pickle.load(file)
    with open("metastasis_status_model.pkl", 'rb') as file:
        metastasis_status_model = pickle.load(file)
    with open("subdivision_model.pkl", 'rb') as file:
        subdivision_model = pickle.load(file)
    with open("metastasis_site_model.pkl", 'rb') as file:
        metastasis_site_model = pickle.load(file)

    # Make predictions
    tumor_stage_predictions = tumor_stage_model.predict(x_test)
    print('Tumor Stage:', tumor_stage_predictions)
    print(type(tumor_stage_predictions))
    if tumor_stage_predictions == 0:
        print('Patient is Tumor Free')
    else:
        metastasis_status_predictions = metastasis_status_model.predict(x_test)
        print('Metastasis Status:', metastasis_status_predictions)
        print(type(metastasis_status_predictions))
        # print(type(metastasis_status_predictions))
        # print('Tumor Stage:', tumor_stage_predictions)
        # if metastasis_status_predictions == 0:
        #     print('M0: No metastasis')
        #     print('Genes of interest:')
            # tumor_stage_weights = tumor_stage_model.coef_
            # print(tumor_stage_weights)
            #
            # weights = np.array(tumor_stage_weights)
            # cpg_sites = ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767',
            #              'cg18766900', 'cg20253855', 'cg21113446', 'cg22300566']
            # abs_weights = np.abs(weights)
            # for class_index, class_weights in enumerate(weights):
            #     print(f'Class {class_index + 1} Influence:')
            #     cpg_influence = dict(zip(cpg_sites, class_weights))
            #     sorted_cpg_influence = sorted(cpg_influence.items(), key=lambda item: abs(item[1]), reverse=True)
            #
            #     for cpg, weight in sorted_cpg_influence:
            #         influence = 'Hypermethylation' if weight > 0 else "Hypomethylation"
            #         print(f' CpG Site: {cpg}, Weight: {weight:.4f}, Influence: {influence}')
            # beta_values = np.array(
            #     [0.975615, 0.688173, 0.631124, 0.77455, 0.857849, 0.751293, 0.717395, 0.5804, 0.52664, 0.759502])
            # decision_function = np.dot(weights, beta_values)
            # predicted_class = np.argmax(decision_function) + 1
            # print(f'Predicted Class: {predicted_class}')
            #
            # coefficients = tumor_stage_weights
            # intercepts = tumor_stage_model.intercept_
            # print(coefficients.shape)
            # print(intercepts.shape)
            #
            # for i in range(coefficients.shape[0]):
            #     plt.bar(range(coefficients.shape[1]), coefficients[i], label=f'Class {i + 1}')
            #
            # plt.xlabel('CpG Sites')
            # plt.ylabel('Coefficients')
            # plt.title('SVM Coefficients for Each Class')
            # plt.legend()
            # plt.show()

        if metastasis_status_predictions == 1:
            subdivision_predictions = subdivision_model.predict(x_test)
            print('Anatomic Neoplasm Subdivision:', subdivision_predictions)
            print(type(subdivision_predictions))
            # print('Tumor Stage:', tumor_stage_predictions)
            metastasis_site_predictions = metastasis_site_model.predict(x_test)
            print('Metastasis Site:', metastasis_site_predictions)
            print(type(metastasis_site_predictions))


if __name__ == "__main__":
    # tumor_stage_prediction()
    # metastasis_prediction()
    anatomic_subdivision_prediction()
    # # metastasis_site_prediction()
    # filepath_1 = "/Users/thaarun/PythonProjects/TCGAProject/output/c8e84511-71ce-4938-a374-99980bf6d7ba" \
    #            ".methylation_array.sesame.level3betas.txt"
    # # filepath_2 = "/Users/thaarun/PythonProjects/TCGAProject/output/e67c69a7-6161-4dd5-9efe-9d13e632989a" \
    # #              ".methylation_array.sesame.level3betas.txt"
    #
    # df_temp = pd.read_csv(filepath_1, sep='\t', header=None)
    # df_temp.rename(columns={0: "sites", 1: "beta_value"}, inplace=True)
    # df_temp.set_index('sites', inplace=True)
    # df_temp = df_temp.T
    #
    # cpg_sites = [
    #     'cg00820405', 'cg06282596', 'cg06721601', 'cg12245706',
    #     'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
    #     'cg21113446', 'cg22300566'
    # ]
    #
    # df_filtered_columns = df_temp[cpg_sites]
    # print("\nFiltered DataFrame with specified CpG sites:")
    # print(df_filtered_columns)

    # output = pipeline_model(df_filtered_columns)

    # df_temp = read_csv()
    # df_x = df_temp.drop(columns=['patient_file_name', 'patient_name', 'tumor_stage', 'anatomic_neoplasm_subdivision',
    #                              'metastasis_site'])
    # df_X = df_x.dropna(axis=1, how='any')
    # # print(df_X)
    #
    # df_t_metastatic_status = df_temp[['metastatic_status']]
    # # print(df_t_metastatic_status)
    #
    # df_MX_patients = df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX']
    # # print(df_MX_patients)
    # # print(df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX'].index.values)
    # MX_indexes = df_t_metastatic_status[df_t_metastatic_status['metastatic_status'] == 'MX'].index.values
    # # print(MX_indexes)
    # test_index = df_t_metastatic_status.index[-3]
    # # test_index.reshape
    # # print(test_index)
    #
    # df_top_ten_sites = df_X[
    #     ['cg00820405', 'cg06282596', 'cg06721601', 'cg12245706', 'cg14196395', 'cg17291767', 'cg18766900', 'cg20253855',
    #      'cg21113446', 'cg22300566']]
    #
    # # df_MX_sites = df_top_ten_sites.iloc[MX_indexes]
    # df_test_sites = df_top_ten_sites.iloc[[test_index]]
    # # df_MX_sites.reset_index(drop=True, inplace=True)
    # # # print(df_MX_sites)
    # # # print(df_MX_sites.shape)
    # df_test_sites.reset_index(drop=True, inplace=True)
    # print(df_test_sites)
    # print(df_test_sites.shape)
    #
    # pipeline_model(df_MX_sites)
    # pipeline_model(df_test_sites)
    # pipeline_model(df_filtered_columns)
