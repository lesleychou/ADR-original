import sys

# FYI, this probably shouldn't have an absolute path to Lesley's machine.
sys.path.append("/Users/lesley/ADR-original")

import numpy as np
from common.svpg.svpg import SVPG
import matplotlib.pyplot as plt
from functools import reduce
import operator
import visdom
import torch

# Visdom must be running to run this script.
# Run `visdom` from the command line to set it up.
vis = visdom.Visdom()
assert vis.check_connection()

# Plotting configuration
plt.rcParams.update({'font.size': 14})
PLOT_COLOR = 'red'

# Helper function
def _rescale( value):
    """Rescales normalized value to be within range of env. dimension
    """
    range_min = 8
    range_max = 50
    return range_min + (range_max - range_min) * value

######################
# Hyperparameter setup
######################
nagents=2
nparams=1
svpg_rollout_length=10
SVPG_train_steps=300
temperature_param=1
# both seed = 101/102 worked well
random_seed=111
torch.manual_seed(random_seed)
np.random.seed(random_seed)


######################
# SVPG initialization
######################
svpg = SVPG( nagents=nagents,
             nparams=nparams,
             max_step_length=0.1,
             svpg_rollout_length=svpg_rollout_length,
             svpg_horizon=1000,
             # change temperature seems have no effect
             temperature=temperature_param,
             discrete=False,
             kld_coefficient=0.01)
#svpg_rewards = np.ones((nagents, 1, nparams))
#print(svpg_rewards)
new_svpg_rewards = np.ones((nagents, 1, nparams))
all_params=[]
rewards=[]
testing_epochs = []
critic_loss = []
current_paras = svpg.step()
current_paras = np.ones( (nagents, svpg_rollout_length, nparams) ) * -1

######################
# Training Loop
######################
for i in range(SVPG_train_steps):
    # FYI, I think this if check doesn't do anything and should likely be removed for clarity.
    # Since the code iterates through the range of training steps above, there's no need
    # to actually check it is less than the training steps and then increment manually.
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
                # if param > 11 and param < 17:
                #     new_svpg_rewards[x][0][0] += 5
                # elif param >= 50:
                #     new_svpg_rewards[x][0][0] -= 50
                # elif param <= 8 or param >= 50:
                #     new_svpg_rewards[x][0][0] -= 10
                # elif param <= 20:
                #     # reward is 100 at 40
                #     #            90 at 41 or 39 .... and so on
                #     #            80 at 42 or 38 .... and so on
                #     #reward = abs(10 - abs(param - 10))*10
                #     new_svpg_rewards[x][0][0] += 2
                # else:
                #     new_svpg_rewards[x][0][0] += 0

                # if param >= 50:
                #     new_svpg_rewards[x][0][0] -= 100
                # elif param <= 8:
                #     new_svpg_rewards[x][0][0] -= 50
                # else:
                #     # we want the distribution to be around here...
                #     target = 10
                #     reward = abs( target - abs( param - target ) )
                #     # max reward of 25
                #     max_reward = 50
                #     if reward > max_reward:
                #         reward = max_reward
                #
                #     new_svpg_rewards[x][0][0] += reward
                if param <= 8.5 or param >= 49.5:
                    new_svpg_rewards[x][0][0] -= 2000
                elif 20 <= param <= 30:
                    new_svpg_rewards[x][0][0] += 200
                else:
                    new_svpg_rewards[x][0][0] -= 200

        #new_svpg_rewards=np.array([[[0]], [[1]]])
        print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        critic_loss_step = svpg.train(i, simulator_rewards=new_svpg_rewards)

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

        reward_all = new_svpg_rewards.reshape(new_svpg_rewards.shape[0], -1)
        reward_mean = reward_all.mean(axis=0)
        rewards.append(list(reward_mean)[0])
        trace = dict( x=testing_epochs ,y=rewards ,mode="markers+lines" ,type='custom' ,
                      marker={'color': PLOT_COLOR ,'symbol': 104 ,'size': "5"} ,
                      text=["one" ,"two" ,"three"] ,name='1st Trace' )
        layout = dict( title="SVPG rewards " ,
                       xaxis={'title': 'Timestamp'} ,
                       yaxis={'title': 'rewards'} )
        vis._send( {'data': [trace] ,'layout': layout ,'win': 'SVPG rewards'} )

    i += 1


######################
# Report and Plotting
######################
    
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
