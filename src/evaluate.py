from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import StreetFighter

MODEL_PATH = './train/final_model'
LOG_DIR = './logs/'

def evaluate_model(model_path: str, n_episodes: int, render: bool, framestack: int = 4):
    """
    Evaluate a trained PPO model on the StreetFighter environment.
    
    Parameters:
    model_path (str): Path to the trained model.
    n_episodes (int): Number of episodes to evaluate.
    render (bool): Whether to render the environment during evaluation.
    framestack (int): Number of frames to stack (default is 4).
    """
    
    # Initialize the StreetFighter environment
    env = StreetFighter()
    
    # Convert environment into a vectorized format
    env = DummyVecEnv([lambda: env])
    
    # Apply frame stacking for better temporal awareness
    env = VecFrameStack(env, framestack, channels_order="last")
    
    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Run evaluation for the specified number of episodes
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)  # Get action from the model
            obs, reward, done, info = env.step(action)  # Take step in the environment
            total_reward += reward  # Accumulate reward
            
            if render:
                env.render()  # Render environment if specified
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Close the environment after evaluation
    env.close()
