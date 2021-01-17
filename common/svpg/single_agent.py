import sys
sys.path.append("/Users/lesley/ADR-original")

import numpy as np
from common.svpg.svpg import SVPG
import matplotlib.pyplot as plt
from functools import reduce
import operator
import visdom

vis = visdom.Visdom()
assert vis.check_connection()

plt.rcParams.update( {'font.size': 14} )
PLOT_COLOR = 'red'


def _rescale( value):
    """Rescales normalized value to be within range of env. dimension
    """
    range_min = 8
    range_max = 50
    return range_min + (range_max - range_min) * value

nagents=2
nparams=1
svpg_rollout_length=2
SVPG_train_steps=500

svpg = SVPG(nagents=nagents ,
            nparams=nparams ,
            max_step_length=0.05,
            svpg_rollout_length=svpg_rollout_length ,
            svpg_horizon=1000 ,
            # change temperature seems have no effect
            temperature=0.1 ,
            discrete=False ,
            kld_coefficient=0.0 )

#svpg_rewards = np.ones((nagents, 1, nparams))
#print(svpg_rewards)

new_svpg_rewards = np.ones((nagents, 1, nparams))

all_params=[]
current_paras = svpg.step()
current_paras = np.ones((nagents,svpg.svpg_rollout_length,svpg.nparams)) * -1
print(current_paras, "------------current_paras intial")
testing_epochs = []
critic_loss = []

for i in range(SVPG_train_steps):
    if i < SVPG_train_steps:
        new_svpg_rewards = np.zeros( (nagents ,1 ,nparams) )
        for t in range( svpg_rollout_length ):
            for x in range(nagents):
                # TODO: the reward logic still have problem:
                #  if the reward is low, output this parameter more,
                #  and next time increase the reward a little bit, because it "trained more on this parameter".
                # [[[31.98860063]
                #   [10]]
                #
                #  [[14.18307286]
                #    [13.19349111]]]
                param = current_paras[x][t]
                # if param <= 8 or param >= 50:
                #     new_svpg_rewards[x][0][0] -= 100
                if param <= 20:
                    # reward is 100 at 40
                    #            90 at 41 or 39 .... and so on
                    #            80 at 42 or 38 .... and so on
                    #reward = abs(10 - abs(param - 10))*10
                    reward = 10
                    new_svpg_rewards[x][0][0] += reward
                else:
                    new_svpg_rewards[x][0][0] -= 20

        #new_svpg_rewards=np.array([[[0]], [[1]]])
        print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        critic_loss_step = svpg.train(simulator_rewards=new_svpg_rewards)

        #print(current_paras, "----------input paras")
        simulation_instances = svpg.step()
        new_paras = _rescale(simulation_instances)
        print(new_paras, "----------output new_paras")
        current_paras = new_paras
        new_svpg_rewards = new_svpg_rewards

        all_params.append(list(new_paras.flatten()))

        # Visdom logs:
        testing_epochs.append(i)
        critic_loss.append(critic_loss_step.tolist())
        trace = dict( x=testing_epochs ,y=critic_loss ,mode="markers+lines" ,type='custom' ,
                      marker={'color': PLOT_COLOR ,'symbol': 104 ,'size': "5"} ,
                      text=["one" ,"two" ,"three"] ,name='1st Trace' )
        layout = dict( title="SVPG critic_loss " ,
                       xaxis={'title': 'Timestamp'} ,
                       yaxis={'title': 'critic_loss'} )
        vis._send( {'data': [trace] ,'layout': layout ,'win': 'SVPG critic_loss'} )

    i += 1

print(all_params, "-------before")
all_params = all_params[-200:]
print(all_params)


plot_params = reduce(operator.concat, all_params)

plt.hist(plot_params, bins=50)
xlims = [8, 50]
plt.xlim(xlims[0], xlims[1])

#plt.plot(all_params, 'o')
plt.ylabel( 'SVPG output' )
plt.xlabel( 'SVPG timestamps' )
plt.title( 'SVPG output parameter changing' )
plt.show()





