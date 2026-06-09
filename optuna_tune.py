import argparse
import os

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from data_loader import fetch_multi_asset_data
from stock_universe import MACRO_TICKERS_RL, TICKERS_TECH_EXPANDED
from trading_env import TaiwanStockEnv
from train_portfolio import build_policy_kwargs

WINDOW_SIZE = 20
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-06-30"


def optimize_ppo(trial: optuna.Trial):
    # 1. Suggest hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 3, 30)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])

    # ensure batch_size <= n_steps
    if batch_size > n_steps:
        batch_size = n_steps

    policy_kwargs = build_policy_kwargs(
        features_dim=features_dim,
        temporal_extractor=True,
        window_size=WINDOW_SIZE
    )

    # 2. Build environments
    global train_data, val_data
    
    train_env = TaiwanStockEnv(
        df_dict=train_data,
        window_size=WINDOW_SIZE,
        topk=5,
        enable_cash_action=True,
    )
    val_env = TaiwanStockEnv(
        df_dict=val_data,
        window_size=WINDOW_SIZE,
        topk=5,
        enable_cash_action=True,
    )

    # 3. Build model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    # 4. Eval callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f"./results_dir/optuna_best_model_{trial.number}",
        log_path="./results_dir/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # 5. Train
    global args
    try:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned() from e

    return eval_callback.best_mean_reward


def load_datasets(overnight_feature_path=None):
    print("Loading train data...")
    train_data = fetch_multi_asset_data(
        tickers=TICKERS_TECH_EXPANDED,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        window_size=WINDOW_SIZE,
        macro_tickers=MACRO_TICKERS_RL,
        overnight_feature_path=overnight_feature_path,
    )
    print("Loading validation data...")
    val_data = fetch_multi_asset_data(
        tickers=TICKERS_TECH_EXPANDED,
        start_date=VAL_START,
        end_date=VAL_END,
        window_size=WINDOW_SIZE,
        macro_tickers=MACRO_TICKERS_RL,
        overnight_feature_path=overnight_feature_path,
    )
    return train_data, val_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--overnight-feature-path",
        default=None,
        help="Optional overnight_gap_features_1d.csv path for RL observation features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(42)
    os.makedirs("results_dir", exist_ok=True)
    
    train_data, val_data = load_datasets(args.overnight_feature_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
