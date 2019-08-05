# Multi-Commander
Multi-agent signal control

Implementation of DQN, Double DQN and Dueling DQN

## usage
### Single agent for single intersection
**Training**

*DQN*
```
python run_rl_control.py --algo DQN --epoch 200 --num_step 2000 --phase_step 1
```
*Double DQN*
```
python run_rl_control.py --algo DDQN --epoch 200 --num_step 2000 --phase_step 1
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --epoch 200 --num_step 2000 --phase_step 1
```

**Inference**

*DQN*
```
python run_rl_control.py --algo DQN --inference --num_step 3000 --ckpt model/DQN_20190803_150924/DQN-200.h5
```
*DDQN*
```
python run_rl_control.py --algo DDQN --inference --num_step 2000 --ckpt model/DDQN_20190801_085209/DDQN-100.h5
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --inference --num_step 2000 --ckpt model/DuelDQN_20190730_165409/DuelDQN-ckpt-10
```

**Simulation**
```
. simulation.sh

open firefox with the url: http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt
```


### Multiple intersections signal control

**Training**

*DQN*
```
python run_rl_multi_control.py --algo MDQN --epoch 1000 --num_step 500 --phase_step 8
```

**Inference**

*DQN*
```
python run_rl_multi_control.py --algo MDQN --inference --num_step 1500 --ckpt model/MDQN_20190804_155422/MDQN-10.h5

python run_rl_multi_control.py --algo MDQN --inference --num_step 1500  --phase_step 2 --ckpt model/MDQN_20190805_120454/MDQN-1.h5
```