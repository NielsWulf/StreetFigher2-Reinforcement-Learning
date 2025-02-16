import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.hyperparameter_opt import run_hyperparameter_optimization

def main():
    parser = argparse.ArgumentParser(description="StreetFighter AI Project")

    # Subcommands for different actions
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the RL model")
    train_parser.add_argument("--model", type=str, default="PPO", help="Model to use for training (default: PPO)")
    train_parser.add_argument("--framestack", type=int, default=4, help="Number of frames to stack (default: 4)")
    train_parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps for training (default: 1,000,000)")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained RL model")
    eval_parser.add_argument("--model_path", type=str, default="./models/best_model_6100000", help="Path to the trained model (default: ./models/best_model_6100000)")
    eval_parser.add_argument("--framestack", type=int, default=4, help="Number of frames to stack during evaluation (default: 4)")
    eval_parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate (default: 10)")
    eval_parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")

    # Optimize subcommand
    opt_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    opt_parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials (default: 50)")

    args = parser.parse_args()

    # Action logic
    if args.action == "train":
        train_model(model_name=args.model, frame_stack=args.framestack, total_timesteps=args.timesteps)
    elif args.action == "evaluate":
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=args.render,
        )
    elif args.action == "optimize":
        run_hyperparameter_optimization(n_trials=args.trials)


if __name__ == "__main__":
    main()
