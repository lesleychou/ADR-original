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
SVPG_train_steps=200

svpg = SVPG(nagents=nagents ,
            nparams=nparams ,
            max_step_length=5 ,
            svpg_rollout_length=svpg_rollout_length ,
            svpg_horizon=25 ,
            # change temperature seems have no effect
            temperature=10.0 ,
            discrete=False ,
            kld_coefficient=0.0 )

#svpg_rewards = np.ones((nagents, 1, nparams))
#print(svpg_rewards)
new_svpg_rewards = np.ones((nagents, 1, nparams))

all_params=[]
current_paras = svpg.step()
current_paras = np.ones((nagents,svpg.svpg_rollout_length,svpg.nparams)) * -1
print(current_paras, "------------current_paras intial")

for i in range(SVPG_train_steps):
    if i < SVPG_train_steps:
        for t in range( svpg_rollout_length ):
            for x in range(nagents):
                #print(new_svpg_rewards[x], "----new_svpg_rewards[x]")
                diff = current_paras[x][t] - 30
                #print(diff, "----diff")
                # TODO: the reward logic still have problem:
                #  if the reward is low, output this parameter more,
                #  and next time increase the reward a little bit, because it "trained more on this parameter".
                if -10 <= diff <= 10:
                    #print("----here")
                    new_svpg_rewards[x][0][0] += 100
                else:
                    #print("----else")
                    new_svpg_rewards[x][0][0] += 0

        #new_svpg_rewards=np.array([[[0]], [[1]]])
        #print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        svpg.train(simulator_rewards=new_svpg_rewards)

        print(current_paras, "----------input paras")
        simulation_instances = svpg.step()
        new_paras = _rescale(simulation_instances)
        print(new_paras, "----------output new_paras")
        current_paras = new_paras
        new_svpg_rewards = new_svpg_rewards

        all_params.append(list(new_paras.flatten()))

    i += 1

print(all_params, "-------all_params")

plt.plot(all_params, 'o')
plt.ylabel( 'SVPG output' )
plt.xlabel( 'SVPG timestamps' )
plt.title( 'SVPG output parameter changing' )
plt.show()





