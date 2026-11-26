# ============================================================================
#   Fully Connected Neural Networks
# ============================================================================
#
#   * Fully connected neural networks (FCNNs) can learn complex non-linear
#     relationships between process features and targets.
#
#   * Network architecture, activation function, and learning rate are
#     important hyperparameters that must be optimized.
#
#   * This example performs a simple grid search search over multiple
#     FCNN architectures and hyperparameter settings.
#
#   * Model performance is evaluated using RMSE and NRMSE on a held-out
#     test set.
#
# ============================================================================

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
import matplotlib.pyplot as plt

from classes import FullyConnectedNeuralNetwork
from functions import progress_bar

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_7factor.csv"))

# Define input features and target variables
features = df.columns[3:]
targets = df.columns[:3]

# Convert DataFrame columns to NumPy arrays
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the hyperparameter search space
param_grid = dict(
    hidden_layers=[
        (5,),
        (10,),
        (10, 10),
        (20, 10),
        (20, 20)
    ],

    activation=[
        "relu",
        "leakyrelu",
        "tanh",
        "sigmoid"
    ],

    learning_rate=[
        1e-2,
        1e-3,
        1e-4
    ],

    batch_size=[
        32,
        128,
        512,
        2048
    ],

    weight_decay=[
        0.0,
        1e-5,
        1e-4,
        1e-3
    ]
)

doHPOpt = False
if doHPOpt:
    # Track iterations for progress bar
    c_iter = 0
    n_iters = len(ParameterGrid(param_grid))

    # Records to export
    columns = list(param_grid.keys()) + ["target", "NRMSE","NRMSE Std. dev", "RMSE","RMSE Std. Dev"]
    records = []

    # Train and evaluate every hyperparameter combination.
    # ParameterGrid enables GridSearchCV-type behaviour.
    for hyperparameters in ParameterGrid(param_grid):

        # Initialize the model
        model = FullyConnectedNeuralNetwork(
            epochs= 100,
            verbose=False,
            random_state=0,
            **hyperparameters
        )

        # Fit the model to the training data
        model.fit(x_train, y_train)

        # Predict the test set targets
        y_pred = model.predict(x_test)

        # Calculate the interquartile range (IQR)
        iqr = np.quantile(y_test, q=0.75, axis=0) - np.quantile(y_test, q=0.25, axis=0)

        # Calculate performance metrics for each target variable
        for j, target in enumerate(targets):
            # Compute RMSE
            rmse = root_mean_squared_error(y_test[:, j], y_pred[:, j])

            # Compute NRMSE using the IQR for normalization
            nrmse = rmse / iqr[j]

            # Compute RMSE Standard Devation
            rmse_values = [root_mean_squared_error(y_test[idx, j], y_pred[idx, j]) for idx in [np.random.choice(len(y_test), len(y_test), replace=True) for _ in range(1000)]]
            rmse_std = np.std(rmse_values)
            nrmse_std = rmse_std / iqr[j]

            record = tuple([hyperparameters[key] for key in columns[:len(hyperparameters)]] + [target, nrmse,nrmse_std, rmse, rmse_std])
            records.append(record)

        c_iter += 1
        progress_bar(c_iter, n_iters)

    df_export = pd.DataFrame.from_records(records, columns=columns)
    df_export.to_csv(os.path.join("tables", "table_grid_search.csv"), index=False)

#
#Following sections does 4-fold validation on optimized model
#

best_params = dict(
        hidden_layers=(20,10),
        activation="sigmoid",
        learning_rate=0.01,
        batch_size=32,
        weight_decay=0.0
    )

doCV = True
if doCV:
    kf = KFold(n_splits=4, shuffle=True, random_state=0)

    records_cv = []
    histories = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):

        print(f"Fold {fold}/4")

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = FullyConnectedNeuralNetwork(epochs=100, verbose=False, random_state=0, **best_params)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        histories.append(dict(
            loss_train=model.loss_train_history.copy(),
            loss_valid=model.loss_valid_history.copy()
        ))

        iqr = np.quantile(y_test, 0.75, axis=0) - np.quantile(y_test, 0.25, axis=0)

        for j, target in enumerate(targets):
            rmse = root_mean_squared_error(y_test[:, j], y_pred[:, j])
            records_cv.append((fold, target, rmse, rmse / iqr[j]))

    df_cv = pd.DataFrame(records_cv, columns=["Fold", "Target", "RMSE", "NRMSE"])

    df_summary = (
        df_cv.groupby("Target")
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            NRMSE_Mean=("NRMSE", "mean"),
            NRMSE_Std=("NRMSE", "std")
        )
        .reset_index()
    )

    df_cv.to_csv(os.path.join("tables", "table_4fold_cv_results.csv"), index=False)
    df_summary.to_csv(os.path.join("tables", "table_4fold_cv_summary.csv"), index=False)

    print(df_summary)

    fig, axes = plt.subplots(2, 2, figsize=(10,8))

    for ax, history, fold in zip(axes.flatten(), histories, range(1,5)):
        ax.plot(history["loss_train"], label="Training Loss")
        ax.plot(history["loss_valid"], label="Validation Loss")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "figure_4fold_learning_curves.png"), dpi=300)
    plt.show()

#
#This section uses 3-fold cross validation to compare effect of hyperparameters of model error
#

#
# Hyperparameter sensitivity analysis (3-fold CV)
#

doSensitivity = True
if doSensitivity:

    sensitivity_grid = {
        "hidden_layers": [
            (5,),
            (10,),
            (20,10), #optimal
            (20,20),
            (50,50)
        ],

        "activation": [
            "sigmoid", #optimal
            "tanh",
            "relu",
            "leakyrelu"
        ],

        "learning_rate": [
            1e-4,
            1e-3,
            1e-2, #optimal
            5e-2,
            1e-1
        ],

        "batch_size": [
            8,
            16,
            32, #optimal
            64,
            128
        ],

        "weight_decay": [
            0.0, #optimal
            1e-6,
            1e-5,
            1e-4,
            1e-3
        ]
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=0)

    records_cv2 = []

    for param_name, test_values in sensitivity_grid.items():

        print(f"\nTesting {param_name}")

        for value in test_values:

            params = best_params.copy()
            params[param_name] = value

            for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):

                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = FullyConnectedNeuralNetwork(
                    epochs=100,
                    verbose=False,
                    random_state=0,
                    **params
                )

                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)

                iqr = np.quantile(y_test, 0.75, axis=0) - np.quantile(y_test, 0.25, axis=0)

                for j, target in enumerate(targets):

                    rmse = root_mean_squared_error(
                        y_test[:, j],
                        y_pred[:, j]
                    )

                    nrmse = rmse / iqr[j]

                    records_cv2.append((
                        param_name,
                        str(value),
                        fold,
                        target,
                        rmse,
                        nrmse
                    ))

    df_sensitivity = pd.DataFrame(
        records_cv2,
        columns=[
            "Hyperparameter",
            "Value",
            "Fold",
            "Target",
            "RMSE",
            "NRMSE"
        ]
    )

    df_sensitivity_summary = (
        df_sensitivity
        .groupby(["Hyperparameter", "Value", "Target"])
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            NRMSE_Mean=("NRMSE", "mean"),
            NRMSE_Std=("NRMSE", "std")
        )
        .reset_index()
    )

    df_sensitivity.to_csv(
        os.path.join(
            "tables",
            "table_hyperparameter_sensitivity_raw.csv"
        ),
        index=False
    )

    df_sensitivity_summary.to_csv(
        os.path.join(
            "tables",
            "table_hyperparameter_sensitivity_summary.csv"
        ),
        index=False
    )

    for param in df_sensitivity_summary["Hyperparameter"].unique():

        temp = df_sensitivity_summary[
            df_sensitivity_summary["Hyperparameter"] == param
        ]

        plt.figure(figsize=(6,4))

        for target in targets:

            target_data = temp[temp["Target"] == target]

            plt.errorbar(
                range(len(target_data)),
                target_data["NRMSE_Mean"],
                yerr=target_data["NRMSE_Std"],
                marker="o",
                capsize=4,
                label=target
            )

        plt.xticks(
            range(len(target_data)),
            target_data["Value"],
            rotation=30
        )

        plt.xlabel(param)
        plt.ylabel("Mean NRMSE")
        plt.title(f"Effect of {param} on Model Error")
        plt.legend()
        plt.grid()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                "figures",
                f"figure_{param}_sensitivity.png"
            ),
            dpi=300
        )

        plt.close()

    print(df_sensitivity_summary)