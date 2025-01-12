import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import StreetFighter

# Initialize Pygame for controller input
pygame.init()
pygame.joystick.init()

# Initialize the first connected joystick
if pygame.joystick.get_count() == 0:
    raise ValueError("No controller detected. Please connect an Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick Name: {joystick.get_name()}")
print(f"Number of Axes: {joystick.get_numaxes()}")
print(f"Number of Buttons: {joystick.get_numbuttons()}")

# Define the mapping of controller inputs to actions
def get_controller_action():
    pygame.event.pump()  # Process input events

    # Initialize action array
    action = [0] * 12

    # Map joystick axes to movement actions
    left_stick_y = joystick.get_axis(1)  # Vertical movement
    left_stick_x = joystick.get_axis(0)  # Horizontal movement

    # Threshold for detecting movement
    threshold = 0.5

    if left_stick_y < -threshold:  # Up
        action[0] = 1
    elif left_stick_y > threshold:  # Down
        action[1] = 1

    if left_stick_x < -threshold:  # Left
        action[2] = 1
    elif left_stick_x > threshold:  # Right
        action[3] = 1

    # Map buttons to specific actions
    if joystick.get_button(0):  # A Button
        action[4] = 1
    if joystick.get_button(1):  # B Button
        action[5] = 1
    if joystick.get_button(2):  # X Button
        action[6] = 1
    if joystick.get_button(3):  # Y Button
        action[7] = 1

    if joystick.get_button(4):  # Left Bumper
        action[8] = 1
    if joystick.get_button(5):  # Right Bumper
        action[9] = 1
    if joystick.get_button(7):  # Start
        action[10] = 1
    if joystick.get_button(6):  # Back
        action[11] = 1

    return action

# Play against the trained RL model
def play_against_model(model_path, n_episodes=5):
    env = StreetFighter()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    model = PPO.load(model_path)
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        print(f"Starting Episode {episode + 1}")

        while not done:
            # Model's action
            model_action, _ = model.predict(obs)
            obs, reward, done, info = env.step(model_action)

            # Render the environment
            env.render()
            
            # Human's action via Xbox controller
            human_action = get_controller_action()

            # Pass human_action wrapped in a list
            obs, reward, done, info = env.step([human_action])

    env.close()

# Example usage
if __name__ == "__main__":
    model_path = "../models/best_model_30000"  # Replace with your model's path
    play_against_model(model_path)
