import math

import torch.optim as optim
from torch.distributions.kl import kl_divergence

from scipy.spatial.distance import squareform, pdist

from common.svpg.particles import SVPGParticle
from common.svpg.svpg_utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
eps = np.finfo(np.float32).eps.item()

LEARNING_RATE = 3e-4
HIDDEN_DIMENSIONS = 100

class SVPG:
    """Class that implements Stein Variational Policy Gradient
    Input is the current randomization settings, and output is either a 
    direction to move in (Discrete - for 1D/2D) or a delta across all parameter (Continuous)
    """
    def __init__(self, nagents, nparams, max_step_length, svpg_rollout_length, svpg_horizon, temperature, discrete, kld_coefficient):
        self.particles = []
        self.prior_particles = []
        self.optimizers = []
        self.max_step_length = max_step_length
        self.svpg_rollout_length = svpg_rollout_length
        self.svpg_horizon = svpg_horizon
        self.temperature = temperature 
        self.nagents = nagents
        self.gamma = 0.99

        self.nparams = nparams
        self.noutputs = nparams * 2 if discrete else nparams
        self.discrete = discrete
        self.kld_coefficient = kld_coefficient

        self.last_states = np.random.uniform(0, 1, (self.nagents, self.nparams))
        self.timesteps = np.zeros(self.nagents)

        for i in range(self.nagents):
            # Initialize each of the individual particles
            policy = SVPGParticle(input_dim=self.nparams,
                                  output_dim=self.noutputs,
                                  hidden_dim=HIDDEN_DIMENSIONS,
                                  discrete=discrete).to(device)

            prior_policy = SVPGParticle(input_dim=self.nparams,
                                  output_dim=self.noutputs,
                                  hidden_dim=HIDDEN_DIMENSIONS,
                                  discrete=discrete,
                                  freeze=True).to(device)

            optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
            self.particles.append(policy)
            self.prior_particles.append(prior_policy)
            self.optimizers.append(optimizer)

    def _squared_dist(self, X):
        # Compute squared distance matrix using torch.pdist
        dists = torch.pdist(X)
        inds_l, inds_r = np.triu_indices(X.shape[-2], 1)
        res = torch.zeros(*X.shape[:-2], X.shape[-2], X.shape[-2], dtype=X.dtype, device=X.device)
        res[..., inds_l, inds_r] = dists
        res[..., inds_r, inds_l] = dists

        return res

    def _Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """

        X_np = X.cpu().data.numpy()
        pairwise_dists = squareform(pdist(X_np))**2

        # Median trick
        h = np.median(pairwise_dists)  
        h = np.sqrt(0.5 * h / np.log(self.nagents+1))

        # Compute RBF Kernel
        Kxx = torch.exp(-torch.from_numpy(pairwise_dists).to(device).float() / h**2 / 2)

        # Compute kernel gradient
        dxKxx = -(Kxx).matmul(X)
        sumKxx = Kxx.sum(1)
        for i in range(X.shape[1]):
            dxKxx[:, i] = dxKxx[:, i] + X[:, i].matmul(sumKxx)
        dxKxx /= (h ** 2)

        return Kxx, dxKxx

    def select_action(self, policy_idx, state):
        # TODO: How does this work???
        # for i in range(10):
        #     state = np.array( [[float(i/10)]] )
        #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #     policy = self.particles[policy_idx]
        #     prior_policy = self.prior_particles[policy_idx]
        #     dist, value = policy(state)
        #     print(state, "-----state")
        #     print(dist, "-------dist")
        #     prior_dist, _ = prior_policy(state)
        #     print(prior_dist, "-------prior-dist")

        state = torch.from_numpy( state ).float().unsqueeze( 0 ).to( device )
        policy = self.particles[policy_idx]
        prior_policy = self.prior_particles[policy_idx]
        # state: input reward
        # value: quality value from critic
        dist ,value = policy( state )
        prior_dist ,_ = prior_policy( state )

        action = dist.sample()

        # log of the pdf/pmf evaluated at "action"
        policy.saved_log_probs.append(dist.log_prob(action))
        # self.particles[i].saved_klds is kl_ds for dist and prior_dist
        policy.saved_klds.append(kl_divergence(dist, prior_dist))
        
        # TODO: Gross temporary hack
        if self.nparams == 1 or self.discrete:
            action = action.item()
        else:
            action = action.squeeze().cpu().detach().numpy()

        # TODO: what is action, what is value?
        return action, value

    def compute_returns(self, next_value, rewards, masks, klds):
        # TODO: in toy example, the return seems only slightly different or same with rewards.
        #  Is this true in ADR? Is the klds messed up?
        '''
        A better A2C with Proper Entropy Bonuses
        :param next_value: _, next_value = self.select_action(i, self.last_states[i])
        :param rewards: particle_rewards
        :param masks:
        :param klds: self.particles[i].saved_klds, kl_ds for dist and prior_dist
        (from different policy),based on the same state
        :return: the array of reversed R
        '''
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            # Eq. 80: https://arxiv.org/abs/1704.06440
            # an A2C proper policy gradient estimators
            # 0/1 * R + (particle_reward - klds of dist and prion_dist)
            # self.kld_coefficient = 0.0
            R = self.gamma * masks[step] * R + (rewards[step] - self.kld_coefficient * klds[step])
            returns.insert(0, R)

        # tensor([[-44.]]) -----after R
        # tensor([[-87.5600]]) -----after R
        # [tensor([[-87.5600]]), tensor([[-44.]]] ----returns
        return returns

    def step(self):
        """Rollout trajectories, starting from random initializations,
        of randomization settings, each of svpg_rollout_length size
        Then, send it to agent for further training and reward calculation
        """
        self.simulation_instances = np.zeros((self.nagents, self.svpg_rollout_length, self.nparams))

        # Store the values of each state - for advantage estimation
        self.values = torch.zeros((self.nagents, self.svpg_rollout_length, 1)).float().to(device)
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.nagents, self.svpg_rollout_length))

        for i in range(self.nagents):
            self.particles[i].reset()
            current_sim_params = self.last_states[i]

            for t in range(self.svpg_rollout_length):
                self.simulation_instances[i][t] = current_sim_params
                action, value = self.select_action(i, current_sim_params)

                self.values[i][t] = value
                clipped_action = action * self.max_step_length # length = 0.05
                # TODO: action is relative action, clipped by [0,1].
                #  so we have a lot of 0/1 output. does this make sense?
                next_params = np.clip(current_sim_params + clipped_action, 0, 1)
                #next_params = np.clip(np.array([clipped_action]), 0, 1)
                # TODO: for non-ADR, should the svpg_horizon be 25?
                if np.array_equal(next_params, current_sim_params) or self.timesteps[i] + 1 == self.svpg_horizon:
                    next_params = np.random.uniform(0, 1, (self.nparams,))
                    # TODO: why when "next_params==current_sim_params", masks is done?
                    self.masks[i][t] = 0 # done = True
                    self.timesteps[i] = 0

                current_sim_params = next_params
                #self.simulation_instances[i][t] = current_sim_params
                self.timesteps[i] += 1

            self.last_states[i] = current_sim_params

        return np.array(self.simulation_instances)

    def train(self, simulator_rewards):
        """Train SVPG agent with rewards from rollouts
        """
        policy_grads = []
        parameters = []

        for i in range(self.nagents):
            policy_grad_particle = []
            # Calculate the value of last state - for Return Computation
            # TODO: what is self.last_states?
            #  start with random value in [0, 1], it represents last_SVPG_output

            _, next_value = self.select_action(i, self.last_states[i])

            particle_rewards = torch.from_numpy(simulator_rewards[i]).float().to(device)
            masks = torch.from_numpy(self.masks[i]).float().to(device)

            # Calculate entropy-augmented returns, advantages
            returns = self.compute_returns(next_value, particle_rewards, masks, self.particles[i].saved_klds)

            returns = torch.cat(returns).detach()

            # advantages = Q(S,a) - V(s): another version of Q-value with lower variance
            advantages = returns - self.values[i]

            # logprob * A = policy gradient (before backwards)
            # dist.log_prob(action) from the def select_action()
            for log_prob, advantage in zip(self.particles[i].saved_log_probs, advantages):
                policy_grad_particle.append(log_prob * advantage.detach())

            # Compute value loss, update critic
            self.optimizers[i].zero_grad()
            critic_loss = 0.5 * advantages.pow(2).mean()
            critic_loss.backward(retain_graph=True)
            self.optimizers[i].step()

            # Store policy gradients for SVPG update
            self.optimizers[i].zero_grad()
            policy_grad = -torch.cat(policy_grad_particle).mean()
            policy_grad.backward()

            # Vectorize parameters and PGs
            # TODO: why to do this here?
            vec_param, vec_policy_grad = parameters_to_vector(
                self.particles[i].parameters(), both=True)
            # make the 1-dim vector into a length=1 2-dim vector
            policy_grads.append(vec_policy_grad.unsqueeze(0))
            parameters.append(vec_param.unsqueeze(0))

        # calculating the kernel matrix and its gradients
        parameters = torch.cat(parameters)
        Kxx, dxKxx = self._Kxx_dxKxx(parameters)
        policy_grads = 1 / self.temperature * torch.cat(policy_grads)

        grad_logp = torch.mm(Kxx, policy_grads)
        grad_theta = (grad_logp + dxKxx) / self.nagents

        # explicitly deleting variables does not release memory :(
        #print(grad_theta, "------grad_theta")
        # update param gradients
        for i in range(self.nagents):
            vector_to_parameters(grad_theta[i],
                 self.particles[i].parameters(), grad=True)
            self.optimizers[i].step()

    def save(self, directory):
        for i in range(self.nagents):
            torch.save(self.particles[i].state_dict(), '{}/particle_{}.pth'.format(directory, i))
            torch.save(self.particles[i].critic.state_dict(), '{}/particle_critic{}.pth'.format(directory, i))

    def load(self, directory):
        return # TODO: Does this solve the value function issue?
        for i in range(self.nagents):
            prior = torch.load('{}/particle_{}.pth'.format(directory, i))
            particle = self.particles[i].state_dict()
            actor = {k: v for k, v in prior.items() if k.find('critic') == -1}

            # Only load the actor!
            self.particles[i].load_state_dict(actor, strict=False)

    def load_prior_particles(self, directory):
        for i in range(self.nagents):
            self.prior_particles[i].load_state_dict(torch.load('{}/particle_{}.pth'.format(directory, i), map_location=device))
            self.prior_particles[i].freeze()

    def _process_action(self, action):
        """Transform policy output into environment-action
        """
        if self.discrete:
            if self.nparams == 1:
                if action == 0:
                    action = [-1.]
                elif action == 1:
                    action = [1.]
            elif self.nparams == 2:
                if action == 0:
                    action = [-1., 0]
                elif action == 1:
                    action = [1., 0]
                elif action == 2:
                    action = [0, -1.]
                elif action == 3:
                    action = [0, 1.]
        else:
            if isinstance(action, float):
                action = np.clip(action, -1, 1)
            else:
                action /= np.linalg.norm(action, ord=2)

        return np.array(action)
