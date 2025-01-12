# **StreetFighter AI: Training and Optimization Framework**

This repository provides a reinforcement learning framework to train and evaluate AI agents to play **Street Fighter II** using the `retro` library. It leverages **Stable-Baselines3** for RL algorithms, **Optuna** for hyperparameter optimization, and TensorBoard for progress visualization.

---

## **Features**
- 🏋️ **Train AI Agents:** Train RL agents from scratch or resume training from existing models.
- 📊 **Evaluate Models:** Evaluate trained agents using custom metrics like rewards, health, and frame deltas.
- 🔧 **Hyperparameter Optimization:** Automatically find optimal hyperparameters for training using **Optuna**.
- 📈 **Training Visualization:** Monitor training progress with **TensorBoard**.

---

## **Setup**

### Prerequisites
Before starting, ensure the following:
- Python 3.8 or lower
- An emulator-compatible ROM for **Street Fighter II**

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/StreetFighter-AI.git
   cd StreetFighter-AI

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv streetfighter
    source streetfighter/bin/activate  # On Windows: streetfighter\Scripts\activate
    pip install -r requirements.txt

3. Import the ROM file:
   
   - Place the ROM file (e.g., StreetFighterIISpecialChampionEdition-Genesis.smc) in the project directory
   - Import it using retro



  
   ```bash
   python -m retro.import .
   ```
---

### **Usage**

#### **1. Training a Model**

  Run the following command:
  ```bash
  python main.py train --framestack 8 --timesteps 1000000
  ```

Resume Training
To continue training from an existing model, use:
```bash
python main.py train --model_path ./models/best_model_6100000.zip --timesteps 500000
```
Arguments:
- **--framestack:** Number of frames to stack for training (default: 8)
- **--timesteps:** Total timesteps for training (default: 1,000,000)
- **--model_path:** Path to an existing model for resuming training (optional)

#### **2. Evaluating a Model**
Evaluate a trained model and observe its performance over a specified number of episodes.

#### Example Command
```bash
python main.py evaluate --model_path ./models/best_model_6100000.zip --episodes 5 --render
```
Arguments:
- **--model_path:** Path to the trained model
- **--episodes:** Number of episodes to evaluate (default: 10)
- **--render:** (Optional) Enable rendering of the environment

#### **3. Hyperparameter Optimization**
Optimize hyperparameters using **Optuna** to fine-tune a model's performance.

#### Example Command
```bash
python main.py optimize --model_path ./models/best_model_6100000.zip --trials 50 --framestack 4 --timesteps 200000
```
Arguments:
- **--model_path:** Path to the trained model (optional, for fine-tuning)
- **--trials:** Number of optimization trials (default: 50)
- **--framestack:** Number of frames to stack (default: 8)
- **--timesteps:** Number of timesteps per trial (default: 200,000)

---

### **Project Structure**

```plaintext
StreetFighter-AI/
├── models/                   # Trained models and checkpoints
├── logs/                     # TensorBoard logs
├── src/                      # Source code
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation logic
│   ├── hyperparameter_opt.py # Hyperparameter optimization logic
│   ├── environment.py        # Custom game environment
│   ├── callbacks.py          # Callbacks for saving models and logs
│   ├── utils.py              # Helper functions
├── notebooks/                # Jupyter notebooks for experimentation
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── main.py                   # Entry point for executing commands
```


---

### **Visualization**

Monitor training progress with **TensorBoard**:
```bash
tensorboard --logdir ./logs
```
Example visualizations include:
- **Episode Rewards:** Monitor the agent's average reward over time.
- **Episode Lengths:** Track how long episodes last as the agent improves

### Observation/Action Space Errors: 
Ensure the framestack parameter matches between training and evaluation:







