# maze_rl
<p align="center">
A multi-agent cooperative maze game using Open AI's gym.
</p>
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Ising_model_5x5_0.svg/2000px-Ising_model_5x5_0.svg.png" width=30%>
</p>

## About
This reinforcement learning (RL) game uses Proximal Policy Optimization (PPO2) to train two agents to cooperate in figuring there way out of a n*m (5x5) grid world (the "Maze"). 

**Rules:**<br>
- Agents must stay within bounds  of the nxm Maze 
- Agents cannot occupy the same position simultaneously 

**Entities:**<br>
- 0: Available Space
- 1: Agent 1 
- 2: Agent 2 
- 3: A Trap (Instant Loss)
- 4: A Teleportation Portal (Moves partner 3 steps North)
- 5: An Exit (A win) 

**Action Space (Discrete):** <br>
- 0: Up
- 1: Right
- 2: Down 
- 3: Left 

**Observation Space:**
- (np.array) -> next player and maze world, flattened, concatenated on horizontal axis.

**State:**
- P: In Play 
- L: Agents Lost 
- W: Agents Won 

**Reward Function/Incentive Mechanism:**
- Reward Range: (-200, 200) 
- Incentives: +1 for Exploring 
- Penalties: -2 for moving to a previopusly visited cell

## Getting started
Clone the repo:
```
https://github.com/codeamt/maze_rl.git
```

Install depemndencies:
```
cd maze_rl && pip install -r requirements.txt
```

## Running Episodes (Local)
Change into src directory and run script:
```
cd code && python main.py
```
### Args: 

- **--epochs** <br>
(type: int, default: 500000)<br>
Number of epochs

- **--lr** <br>
(type: float, default: 0.001)<br>
Learning rate for training policy

- **--gamma** <br>
(type: float, default: 0.000001)<br>
Discount Factor 

- **--lam** <br>
(type; float, default: 0)<br>
Lambda/GAE Factor

- **--world** (type: List[List[int],<br> 
default:<br> 
[[1, 0, 2, 0, 0],<br>
[0, 0, 0, 0, 0],<br> 
[4, 0, 0, 0, 0],<br>
[0, 0, 3, 3, 0],<br>
[0, 0, 3, 5, 0]]

- **--inference** (type: bool, default: True)
Whether or not to run a quick inference test after training to test performance. 


After some warnings from Tensorflow, you will see the updated maze after each steps on the output.
Check the render folder for a training report.


## Docker Build/Run Instructions
Build the image:
```
docker build -t <choose img name> .
```
Run the container: 
```
docker run -d -it --name=<name of this container> <choosen img name> python main.py --epochs=500000 
```

