# Multi-Commander
Multi-agent signal control

Implementation of DQN, Double DQN and Dueling DQN

### usage
**Training**

*DQN*
```
python run_rl_control.py
```
*Double DQN*
```
python run_rl_control.py --algo DDQN --epoch 10 # double DQN
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --epoch 1
```

**Inference**

*DQN*
```
python run_rl_control.py --algo DQN --inference --ckpt model/20190729_163837/dqn-10.h5
```
*DDQN*
```
python run_rl_control.py --algo DDQN --inference --ckpt model/20190729_163837/dqn-10.h5
```

*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --inference --ckpt model/DuelDQN_20190729_163837/model-ckpt ...
```

**Simulation**
```
. simulation.sh
open firefox (http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt)
```


