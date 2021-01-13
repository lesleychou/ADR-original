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
svpg_rollout_length=5
SVPG_train_steps=500

svpg = SVPG(nagents=nagents ,
            nparams=nparams ,
            max_step_length=5 ,
            svpg_rollout_length=svpg_rollout_length ,
            svpg_horizon=25 ,
            # change temperature seems have no effect
            temperature=10.0 ,
            discrete=False ,
            kld_coefficient=0.0 )

svpg_rewards = np.ones((nagents, svpg_rollout_length, nparams))
new_svpg_rewards = np.ones((nagents, svpg_rollout_length, nparams))

all_params=[]
simulation_instances = svpg.step()

for i in range(SVPG_train_steps):
    if i < SVPG_train_steps:
        #print(simulation_instances, "-------simulation_instances")
        for t in range(svpg_rollout_length):
            for i in range(nagents):
                rewrd = svpg_rewards[i][t] - 40
                if -2 <= rewrd <= 2:
                    new_svpg_rewards[i][t] = -10000
                else:
                    new_svpg_rewards[i][t] = rewrd

        #new_svpg_rewards=np.array([[[0]], [[1]]])
        print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        svpg.train(simulator_rewards=new_svpg_rewards)

        simulation_instances = svpg.step()
        new_paras = _rescale(simulation_instances)
        print(new_paras, "----------new_paras")
        svpg_rewards = new_paras

        all_params.append(list(new_paras.flatten()))

    i += 1

print(all_params, "-------all_params")

plt.plot(all_params, 'o')
plt.ylabel( 'SVPG output' )
plt.xlabel( 'SVPG timestamps' )
plt.title( 'SVPG output parameter changing' )
plt.show()
