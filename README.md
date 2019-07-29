# Multi-Commander
Multi-agent signal control

Implementation of DQN, and it's variants

### usage
training
```
python run_rl_control.py # DQN
python run_rl_control.py --algo DDQN --epoch 10 # double DQN

```
inference
```
python run_rl_control.py --inference --ckpt model/20190729_163837/dqn-10.h5
```
simulation
```
. simulation.sh
open firefox (http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt)
```


