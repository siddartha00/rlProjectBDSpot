## Getting Started

### Prerequisites

Make sure you have Python installed (tested with Python 3.11+ recommended). You may want to use a conda environment:

```bash
conda create -n hrlproject python=3.11 -y
conda activate hrlproject
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### Running the Project

1. Navigate to the project folder:

```bash
cd rlProjectBDSpot
```

2. Run the testing script:

```bash
python train_hrl.py
```

> **Note:** In `train_hrl.py`, make sure the following is active:  
> ```python
> if __name__ == "__main__":
>     # Train the policy
>     #train_hrl_policy_dqn()
>     
>     # Test the policy
>     test_hrl_policy()
> ```  
> We are using `test_hrl_policy()` to test the trained policy.

---

### Project Output

<img width="1440" height="245" alt="image" src="https://github.com/user-attachments/assets/64eab67f-2627-42ae-bb8b-c1d2593554d3" />

---

### Project Videos (Straight, Turn, Meta policy)
https://github.com/user-attachments/assets/6c5c689f-8177-4a6a-985f-5ca1d923740c


https://github.com/user-attachments/assets/792b95ad-f1ae-4eaf-b010-d5c5f24f6312


https://github.com/user-attachments/assets/9607fae4-ebb1-4126-9e9c-235cb3a06b2c


---
