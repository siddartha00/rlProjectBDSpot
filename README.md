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

Add an image showing the working output here:

<img width="1440" height="245" alt="image" src="https://github.com/user-attachments/assets/64eab67f-2627-42ae-bb8b-c1d2593554d3" />

---
