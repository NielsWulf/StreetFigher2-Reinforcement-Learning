import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import StreetFighter

LOG_DIR = './logs/'
OPT_DIR = './opt/'

def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 2e-7, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.3),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 0.99)
    }

def run_hyperparameter_optimization():
    def optimize_agent(trial):
        model_params = optimize_ppo(trial)
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 8, channels_order='last')

        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=100000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
        env.close()

        return mean_reward

    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=100, n_jobs=1)
