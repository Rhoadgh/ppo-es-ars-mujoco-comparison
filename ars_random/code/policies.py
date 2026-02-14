'''
Policy class for computing action from weights and observation vector. 
Updated for Gymnasium/MuJoCo v4 compatibility.
'''

import numpy as np
from filter import get_filter

class Policy(object):
    def __init__(self, policy_params):
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        # Ensure the incoming weights match the internal shape
        self.weights[:] = new_weights.reshape(self.weights.shape)[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        # 1. Cast for precision
        if ob.dtype != np.float64:
            ob = ob.astype(np.float64)

        # 2. V2 Normalization Logic (The Whitening)
        # The filter must return: (ob - mean) / std
        ob = self.observation_filter(ob, update=self.update_filter)

        # 3. Compute linear action
        action = np.dot(self.weights, ob)

        # 4. Clipping (Crucial for MuJoCo environments)
        # Most environments expect actions between [-1, 1]
        return np.clip(action, -1.0, 1.0)

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>.
    """
    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        # Initialize weights with float64 to match the deltas table precision
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        # Gymnasium v4 observations are often float32; cast to float64 for precision
        if ob.dtype != np.float64:
            ob = ob.astype(np.float64)

        # Apply the observation filter (MeanStdFilter or NoFilter)
        ob = self.observation_filter(ob, update=self.update_filter)

        # Compute the action: y = Wx
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        # Ensure we return a structured array or list for saving to .npz
        return {'weights': self.weights, 'mu': mu, 'std': std}