import sys

# FYI, this probably shouldn't have an absolute path to Lesley's machine.
sys.path.append("/Users/lesley/ADR-original")
import os
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

# Helper functions
def _rescale(value):
    """Rescales normalized value to be within range of env. dimension
    """
    range_min = 8
    range_max = 500
    return range_min + (range_max - range_min) * value

def plot(params, filename):
    """Plots and saves historgram of values.
    """
    print("Saving plot to: " + filename)
    plt.hist(params, bins=492)
    xlims = [8, 500]
    plt.xlim(xlims[0], xlims[1])
    plt.ylabel('SVPG output count')
    plt.xlabel('sampled range')
    plt.title('SVPG sampled dist')
    plt.savefig(filename, dpi=100)
    plt.close()

######################
# Hyperparameter setup
######################
nagents=2
nparams=1
svpg_rollout_length=10
SVPG_train_steps=500
temperature_param=0.1
# both seed = 101/102 worked well
# random_seed=111
# random_seeds = []
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)


######################
# Training Loop
######################
def train(seed):
    ######################
    # SVPG initialization
    ######################
    svpg = SVPG( nagents=nagents,
                 nparams=nparams,
                 max_step_length=0.04,
                 svpg_rollout_length=svpg_rollout_length,
                 svpg_horizon=1000,
                 # change temperature seems have no effect
                 temperature=temperature_param,
                 discrete=False,
                 kld_coefficient=0.01,
                 load=True)
    #svpg_rewards = np.ones((nagents, 1, nparams))
    #print(svpg_rewards)
    new_svpg_rewards = np.ones((nagents, 1, nparams))
    all_params=[]
    rewards=[]
    testing_epochs = []
    critic_loss = []
    current_paras = svpg.step()
    current_paras = np.ones( (nagents, svpg_rollout_length, nparams) ) * -1

    ####################
    # Main loop
    ####################    
    for i in range(SVPG_train_steps):
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
                # Trained for 1000 epoch, plot the last 200
                if param <= 8.5 or param >= 499.5:
                    new_svpg_rewards[x][0][0] -= 2000
                # elif param >= 480:
                #     new_svpg_rewards[x][0][0] -= 1000
                # # elif param >= 460:
                # #     new_svpg_rewards[x][0][0] -= 500
                elif 100 <= param <= 200:
                    new_svpg_rewards[x][0][0] += 200
                else:
                    new_svpg_rewards[x][0][0] -= 200
                    #new_svpg_rewards=np.array([[[0]], [[1]]])
        #print(new_svpg_rewards, "----------new_svpg_rewards", '\n')
        critic_loss_step = svpg.train(i, simulator_rewards=new_svpg_rewards)

        #print(current_paras, "----------input paras")
        simulation_instances = svpg.step()
        new_paras = _rescale(simulation_instances)
        print(new_paras, "----------output new_paras")
        current_paras = new_paras
        new_svpg_rewards = new_svpg_rewards

        log_param= list(new_paras.flatten())
        RLMPC_LOG = '/Users/lesley/ADR-original/results/distlog'
        dist_path = os.path.join( RLMPC_LOG ,'param' )
        dist_file = open( dist_path ,'a' ,1 )
        dist_file.write( str( log_param ) + '\n' )

        all_params.append(list(new_paras.flatten()))

        # Visdom logs:
        testing_epochs.append(i)
        critic_loss.append(critic_loss_step.tolist())
        trace = dict( x=testing_epochs ,y=critic_loss ,mode="markers+lines" ,type='custom' ,
                      marker={'color': PLOT_COLOR ,'symbol': 104 ,'size': "5"} ,
                      text=["one" ,"two" ,"three"] ,name='1st Trace' )
        window = "SVPG critic_loss " + str(seed)
        layout = dict( title=window,
                       xaxis={'title': 'Timestamp'} ,
                       yaxis={'title': 'critic_loss'} )
        vis._send( {'data': [trace] ,'layout': layout ,'win': window} )

        reward_all = new_svpg_rewards.reshape(new_svpg_rewards.shape[0], -1)
        reward_mean = reward_all.mean(axis=0)
        rewards.append(list(reward_mean)[0])
        trace = dict( x=testing_epochs ,y=rewards ,mode="markers+lines" ,type='custom' ,
                      marker={'color': PLOT_COLOR ,'symbol': 104 ,'size': "5"} ,
                      text=["one" ,"two" ,"three"] ,name='1st Trace' )
        window = "SVPG rewards " + str(seed)        
        layout = dict( title=window,
                       xaxis={'title': 'Timestamp'} ,
                       yaxis={'title': 'rewards'} )
        vis._send( {'data': [trace] ,'layout': layout ,'win': window} )

    ######################
    # Report and Plotting
    ######################
    #print(all_params, "-------before")
    all_params = np.array(all_params)

    y1 = all_params[: ,0:10]
    y2 = all_params[: ,10:20]

    line1 = np.reshape( y1 ,[1 ,-1] ).squeeze()
    print( line1 )
    line2 = np.reshape( y2 ,[1 ,-1] ).squeeze()
    print( line2 )
    # plot_params = reduce(operator.concat, all_params)
    # plot_filename = 'results/' + str(seed) + '.png'
    # plot(plot_params, plot_filename)

    # cmap = plt.cm.jet
    #
    # for i ,y in enumerate( line1 ):
    #     plt.plot( y ,color=cmap( i / float( 10 ) ), label='particle 1')

    plt.plot(line1, color='blue')
    plt.plot(line2, color='red')
    plt.legend()

    plt.show()


######################
# MAIN
######################

# Run on a range of random seeds for robustness.
for i in range(102, 103):
    print("Running on RANDOM SEED: ", str(i))
    torch.manual_seed(i)
    np.random.seed(i)
    train(i)
