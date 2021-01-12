import sys
sys.path.append("/Users/lesley/ADR-original")

import numpy as np
from common.svpg.svpg import SVPG
import matplotlib.pyplot as plt

plt.rcParams.update( {'font.size': 14} )


def _rescale( value):
    """Rescales normalized value to be within range of env. dimension
    """
    range_min = 8
    range_max = 50
    return range_min + (range_max - range_min) * value

nagents=2
nparams=1
svpg_rollout_length=2
SVPG_train_steps=1000

svpg = SVPG(nagents=2 ,
            nparams=1 ,
            max_step_length=5 ,
            svpg_rollout_length=2 ,
            svpg_horizon=25 ,
            temperature=2.0 ,
            discrete=False ,
            kld_coefficient=0.0 )

svpg_rewards = np.ones((nagents, svpg_rollout_length, nparams))
new_svpg_rewards = np.ones((nagents, svpg_rollout_length, nparams))

all_params=[]

for i in range(SVPG_train_steps):
    if i < SVPG_train_steps:
        simulation_instances = svpg.step()

        for t in range(svpg_rollout_length):
            for i in range(nagents):
                rewrd = svpg_rewards[i][t] - 45
                if rewrd <= 0:
                    new_svpg_rewards[i][t] = rewrd
                else:
                    new_svpg_rewards[i][t] = -rewrd

        #print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        svpg.train(simulator_rewards=new_svpg_rewards)

        #new_simulation_instances = svpg.step()
        new_paras = _rescale(simulation_instances)
        #print(new_paras, "----------new_paras")
        svpg_rewards = new_paras

        all_params.append(list(new_paras.flatten()))

    i += 1

print(all_params, "-------all_params")

plt.plot(all_params, 'o')
plt.ylabel( 'SVPG output' )
plt.xlabel( 'SVPG timestamps' )
plt.title( 'SVPG output parameter changing' )
plt.show()
