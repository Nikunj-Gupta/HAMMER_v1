## Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run:
MultiWalker:
```bash
# source .venv/bin/activate
python3 multiagent
```

## File descriptions:
- ppo.py : Contains implementation of PPO for discrete action space environments.
- ppo_cont.py : Contains implementation of PPO for continuous action space environments.
- Cooperative Navigation environment:
  - pretrained_global_agent.py : 
  - random_message.py : 
  - ppo_shared_network :
  - ppo_shared_network_random_message.py : 
  - ppo_shared_network_centralised_training.py : 
  - main_complete_state_and_prev_actions.py : 
  - centralised_training.py : 
- Multi Walker environment:
  - play.py : 
  - multiwalker.py : 