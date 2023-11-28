import optuna
import yaml

# Load our optuna config from config file
cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)['optuna']
NUM_TRIALS = cfg['num_trials']

sampler = cfg['sampler']
if sampler == 'tpe':
    SAMPLER = optuna.samplers.TPESampler
else:
    SAMPLER = None

pruner = cfg['pruner']
if pruner == 'hyperband':
    PRUNER = optuna.pruners.HyperbandPruner
else:
    PRUNER = None

# Create the study
study = optuna.create_study(
    direction="maximize",
    sampler=SAMPLER(),
    pruner=PRUNER()
)


# Responsible for generating all the h-params we want to optimize
def generate_trial_suggestion(trial):
    param_cfgs = cfg['params']
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
def build_model_and_evaluate(optuna_suggestions, model_type="encoder"):
    print(optuna_suggestions)
    if model_type == 'encoder':
        # Build encoder model and evaluate
        # TODO: Add logic for creating model, and add training loop + evaluation
        return 100


def objective_function(trial):
    suggestions = generate_trial_suggestion(trial=trial)
    acc = build_model_and_evaluate(optuna_suggestions=suggestions, model_type=cfg['model'])
    return acc


if __name__ == "__main__":
    study.optimize(objective_function, n_trials=NUM_TRIALS)
    print(f"Optimal Values Found in {NUM_TRIALS} trials:")
    print("-------------------------------------------------")
    for param, optimum_val in study.best_trial.params.items():
        print(f"{param} : {optimum_val}")
