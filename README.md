# Multi-Commander
Multi-agent traffic signal control for DeeCamp2019

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

*QMIX (based on Ray)*
```
python ray_multi_agent.py
```

<img src=demos/QMIX_tensorboard.png />

*MDQN (manul implementation)*
```
python run_rl_multi_control.py --algo MDQN --epoch 1000 --num_step 500 --phase_step 10
```

*MDQN (based on Ray)*
```
python ray_multi_dqn.py
```

**Inference**

*MDQN (manul implementation)*
```
python run_rl_multi_control.py --algo MDQN --inference --num_step 1500 --phase_step 15 --ckpt model/XXXXXXX/MDQN-1.h5
```

*MDQN (based on Ray) (in lab linux)*
```
python ray_multi_dqn_rollout.py --run DQN --checkpoint ~/ray_results/DQN_cityflow_multi_2019-08-11_00-44-52khzt8bnq/checkpoint_400/checkpoint-400 --env cityflow_multi --steps 1000
```



#### Rule based
*1\*6 roadnet*

Generate checkpoint
```
python run_rl_multi_control.py --algo MDQN --epoch 1 --num_step 1 --phase_step 10
```

Generate replay file
```
python run_rl_multi_control.py --algo MDQN --inference --num_step 130 --phase_step 30 --ckpt model/MDQN_20190809_134734/MDQN-1.h5
```


Simulation
```
. simulation.sh

open firefox with the url: http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt
```


#### Installation
*Ciryflow deecamp branch*
```
git clone -b deecamp https://github.com/zhc134/CityFlow.git
pip install .
```

---

<img src=demos/1_6_700/demo_1_6.gif />

<!-- <img src=demos/demo_1_1.gif /> -->