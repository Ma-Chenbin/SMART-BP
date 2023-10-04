import h2o
from h2o.automl import H2OAutoML

# Load PPG-BP data into h2o frame
h2o.init()
data = h2o.import_file('/path/to/BP_interval_norm.csv')

# Split data into training and validation sets
train, valid = data.split_frame(ratios=[0.8], seed=2023)

# Specify the training/validation columns and target variable
x = train.columns[:-1]  # all columns except the last one
y = train.columns[-1]   # the last column is the target BP value

# Initialize AutoML with BP estimation specific parameters
aml = H2OAutoML(max_models=20,
                seed=1,
                include_algos=["GLM", "GBM", "XGBoost", "NaiveBayes", "DRF", "DeepLearning",
                               "StackedEnsemble", "StackedEnsemble_AllModels"],
                max_runtime_secs=3600,
                stopping_metric="RMSE",
                objectif="mse",
                sort_metric="mse",
                keep_cross_validation_predictions=True,
                nfolds=5,
                fold_column="time",
                time_column="time",
                max_cols=80,
                sample_rate=0.8,
                col_sample_rate=0.8,
                balance_classes=False,
                export_checkpoints_dir='/path/to/checkpoints')

aml.estimator_parameters = {
    "GLM": {"ntrees": 100, "alpha": 100, "lambda": 6},
    "GBM": {"ntrees": 100, "max_depth": 6, "min_rows": 10, "learn_rate": 0.001},
    "XGBoost": {"ntrees": 100, "max_depth": 6, "min_rows": 10, "sample_rate": 0.8, "col_sample_rate": 0.8},
    "DRF": {"ntrees": 100, "max_depth": 6, "min_rows": 10, "mtries": 5},
    "DeepLearning": {"epochs": 200, "hidden": [5, 2, 2, 1], "activation": "relu"},
                            }

# Train the model on the training set
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

# Get the best model
best_model = aml.leader

# Evaluate the model on the validation set
perf = best_model.model_performance(valid)
print(perf.rmse())

# Save the model
model_path = h2o.save_model(model=best_model, path='/path/to/save/model', force=True)
print(f"Model saved to {model_path}")
