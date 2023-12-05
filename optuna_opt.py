import optuna
import yaml
from utils import masked_mse, masked_cross_entropy, load_df_with_secondary_struct, test, train
from torch.utils.data import DataLoader
from dataset import RNAInputDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
from models import Encoder

# Load our optuna config from config file
cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)
optuna_cfg = cfg['optuna']
NUM_TRIALS = optuna_cfg['num_trials']
EPOCHS = optuna_cfg['train_epochs']
BATCH_SIZE = optuna_cfg['train_batch_size']

# Data loading config
VAL_PROP = cfg['data']['val_prop']
PRETRAIN = cfg['pretrain']
DEVICE = cfg['device']
SEQ_LENGTH = cfg['data']['seq_length']
EXPERIMENT = cfg['data']['experiment']

# Optuna Sampler and Pruner config
sampler = optuna_cfg['sampler']
if sampler == 'tpe':
    SAMPLER = optuna.samplers.TPESampler
else:
    SAMPLER = None

pruner = optuna_cfg['pruner']
if pruner == 'hyperband':
    PRUNER = optuna.pruners.HyperbandPruner
else:
    PRUNER = None

# Create the study
study = optuna.create_study(
    direction="minimize",
    sampler=SAMPLER(),
    pruner=PRUNER()
)

# Load train data
df_raw = pd.read_csv(cfg['data']['paths']['df'])
df_exp = df_raw[df_raw['experiment_type'] == EXPERIMENT]

if PRETRAIN:
    secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
    df = load_df_with_secondary_struct(df_exp, secondary_struct_df)
else:
    df = df_exp

df_train, df_val = train_test_split(df, test_size=VAL_PROP)
ds_train = RNAInputDataset(df_train, pretrain=PRETRAIN, seq_length=SEQ_LENGTH, snr_filter=True, device=DEVICE, test=False)
ds_val = RNAInputDataset(df_val, pretrain=PRETRAIN, seq_length=SEQ_LENGTH, snr_filter=False, device=DEVICE, test=True)
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)


def generate_trial_suggestion(trial) -> dict:
    """
        Generate trial suggestions for optuna hyperparameters tuning based on the yml config file.

        Parameters:
        - trial: An Optuna Trial object

        Returns:
        - trial_suggestions: A dictionary containing suggested hyperparameter values for the trial.
    """

    param_cfgs = optuna_cfg['params']
    trial_suggestions = {}

    for param_cfg in param_cfgs:
        if param_cfg['type'] == 'int':
            # Suggest an int
            trial_suggestions[param_cfg['name']] = trial.suggest_int(param_cfg['name'], param_cfg['range_min'],
                                                                     param_cfg['range_max'])

        elif param_cfg['type'] == 'float':
            # Suggest a float
            trial_suggestions[param_cfg['name']] = trial.suggest_float(param_cfg['name'], param_cfg['range_min'],
                                                                       param_cfg['range_max'])

        elif param_cfg['type'] == 'bool':
            # Suggest a bool
            trial_suggestions[param_cfg['name']] = trial.suggest_categorical(param_cfg['name'], [True, False])

    return trial_suggestions


# Builds a new model given a set of h-param values and evaluates the model
def build_model_and_evaluate(optuna_suggestions, model_type="encoder") -> float:
    """
       Build a model with the given Optuna suggestions,
       train the model, and evaluate its performance on the validation set.

       Parameters:
       - optuna_suggestions: A dictionary containing hyperparameter values from generate_trail_suggestion function.
       - model_type: A string indicating the type of model to build ('encoder' or 'attention').

       Returns:
       - avg_val_loss: The average validation loss of the trained model.
    """

    model = None
    if model_type == 'encoder':
        weights = None if PRETRAIN else cfg['model']['weights']
        model_cfg = cfg['model'][model_type]
        embedding_cfg = cfg['model']['embedding_cfg']
        model = Encoder(
            embedding_cfg=embedding_cfg,
            num_layers=model_cfg['num_layers'],
            layer_cfg=model_cfg['layer_cfg'],
            seq_length=SEQ_LENGTH,
            weights=weights,
            num_frozen_layers=optuna_suggestions['num_frozen_layers']
        )
    else:
        # TODO: build a new attention model with optuna suggestions
        model = None

    # Train the model
    best_val_loss = 1000000
    for epoch in range(EPOCHS):
        train(
            model=model,
            data_loader=train_loader,
            loss_fn=masked_mse if not PRETRAIN else masked_cross_entropy,
            optimizer=optim.Adam(model.parameters(), lr=optuna_suggestions['learning_rate']),
            device=DEVICE
        )

        # Evaluate model
        avg_val_loss = test(
            model=model,
            data_loader=val_loader,
            loss_fn=masked_mse if not PRETRAIN else masked_cross_entropy,
            device=DEVICE
        )
        best_val_loss = avg_val_loss if (avg_val_loss < best_val_loss) else best_val_loss
    return best_val_loss


def objective_function(trial):
    """
       Objective function for the Optuna optimization.

       Parameters:
       - trial: An Optuna Trial object.

       Returns:
       - avg_val_loss: The average validation loss of the model built with the trial suggestions.
    """

    suggestions = generate_trial_suggestion(trial=trial)
    best_val_loss = build_model_and_evaluate(optuna_suggestions=suggestions, model_type=optuna_cfg['model'])
    return best_val_loss


if __name__ == "__main__":
    study.optimize(objective_function, n_trials=NUM_TRIALS)
    print(f"Optimal Values Found in {NUM_TRIALS} trials:")
    print("-------------------------------------------------")
    for param, optimum_val in study.best_trial.params.items():
        print(f"{param} : {optimum_val}")
