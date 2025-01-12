from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import StreetFighter

MODEL_PATH = './train/final_model'
LOG_DIR = './logs/'

def evaluate_model(model_path: str, n_episodes: int, render: bool, framestack: int = 4):

    env = StreetFighter()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, framestack, channels_order="last")

    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if render:
                env.render()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()
