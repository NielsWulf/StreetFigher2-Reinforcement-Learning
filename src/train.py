from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from src.environment import StreetFighter
from src.callbacks import TrainAndLoggingCallback
from src.utils import linear_schedule

def train_model(model_name="PPO", frame_stack=8, total_timesteps=1000000):
    """
    Train a reinforcement learning model using PPO on the StreetFighter environment.
    
    Parameters:
    model_name (str): The RL model to use.
    frame_stack (int): Number of frames to stack for the environment.
    total_timesteps (int): Total number of timesteps for training.
    """
    
    # Directories for logging and saving model checkpoints
    LOG_DIR = './logs/'
    CHECKPOINT_DIR = './train/'
    
    # Initialize the StreetFighter environment
    env = StreetFighter()
    
    # Monitor the environment to log episode rewards, lengths, etc.
    env = Monitor(env, LOG_DIR)
    
    # Convert environment into a vectorized format required for stable-baselines3
    env = DummyVecEnv([lambda: env])
    
    # Apply frame stacking to provide temporal information to the model
    env = VecFrameStack(env, frame_stack, channels_order='last')
    
    # Define learning rate and clip range schedules for training
    learning_rate_schedule = linear_schedule(2.5e-5, 2.5e-7)  # Linear decay of learning rate
    clip_range_schedule = linear_schedule(0.15, 0.025)  # Linear decay of clip range
    
    # Select the model 
    if model_name == "PPO":
        model = PPO(
            "CnnPolicy",  # Use convolutional neural network policy
            env,  # Training environment
            tensorboard_log=LOG_DIR,  # Log training data for visualization in TensorBoard
            verbose=1,  # Print training progress details
            n_steps=2560,  # Number of steps to run per batch
            gamma=0.906,  # Discount factor for future rewards
            learning_rate=2e-7,  # Initial learning rate
            clip_range=0.369,  # Clipping range for PPO's policy update
            gae_lambda=0.891,  # Lambda for Generalized Advantage Estimation (GAE)
        )
    else:
        # Raise an error if an unsupported model type is provided
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Callback function for periodic checkpoint saving during training
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    # Train the model with the specified total timesteps
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the final trained model
    model.save(CHECKPOINT_DIR + "final_model")
    
    # Close the environment to free resources
    env.close()
