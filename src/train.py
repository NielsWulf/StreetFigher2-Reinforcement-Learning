from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from src.environment import StreetFighter
from src.callbacks import TrainAndLoggingCallback
from src.utils import linear_schedule

def train_model(model_name="PPO", frame_stack=8, total_timesteps=1000000):
    

    LOG_DIR = './logs/'
    CHECKPOINT_DIR = './train/'

    env = StreetFighter()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, frame_stack, channels_order='last')

    # Learning rate and clip range schedules
    learning_rate_schedule = linear_schedule(2.5e-5, 2.5e-7)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Select model (currently only PPO is supported)
    if model_name == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            tensorboard_log=LOG_DIR,
            verbose=1,
            n_steps=2560,
            gamma=0.906,
            learning_rate=learning_rate_schedule,
            clip_range=clip_range_schedule,
            gae_lambda=0.891,
            batch_size=512,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(CHECKPOINT_DIR + "final_model")
    env.close()
