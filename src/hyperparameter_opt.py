import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import StreetFighter

# Directories for logging and optimization results
LOG_DIR = './logs/'
OPT_DIR = './opt/'

def optimize_ppo(trial):
    """
    Defines the hyperparameter search space for PPO using Optuna.
    """
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),  # Number of steps to run for each environment per update
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),  # Discount factor for future rewards
        'learning_rate': trial.suggest_loguniform('learning_rate', 2e-7, 1e-4),  # Learning rate for the optimizer
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.3),  # Clipping parameter for PPO
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 0.99)  # Smoothing parameter for Generalized Advantage Estimation (GAE)
    }

def run_hyperparameter_optimization():
    """
    Runs the hyperparameter optimization process using Optuna.
    """
    def optimize_agent(trial):
        """
        Defines the objective function for Optuna, which trains and evaluates the PPO model.
        """
        # Get hyperparameters from the optimization function
        model_params = optimize_ppo(trial)
        
        # Create and wrap the environment
        env = StreetFighter()  # Initialize custom StreetFighter environment
        env = Monitor(env, LOG_DIR)  # Monitor the environment to log training metrics
        env = DummyVecEnv([lambda: env])  # Wrap the environment for vectorized execution
        env = VecFrameStack(env, 4, channels_order='last')  # Stack 4 frames for better perception of movement
        
        # Initialize the PPO model with the optimized parameters
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        
        # Train the model for 100,000 timesteps
        model.learn(total_timesteps=100000)
        
        # Evaluate the trained model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)  # Run evaluation for 20 episodes
        
        # Close the environment to free resources
        env.close()
        
        return mean_reward  # Return the mean reward as the optimization metric

    # Create an Optuna study object to maximize the agent's performance
    study = optuna.create_study(direction='maximize')
    
    # Run the optimization process for 100 trials
    study.optimize(optimize_agent, n_trials=100, n_jobs=1)

