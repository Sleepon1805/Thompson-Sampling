import numpy as np
import numpy.random as rd


class TwoLayerNNEpsilonGreedy:

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 actions,
                 time_horizon,
                 prior_var,
                 noise_var,
                 epsilon_param=0.0,
                 learning_rate=1e-1,
                 num_gradient_steps=1,
                 batch_size=64,
                 lr_decay=1,
                 leaky_coeff=0.01):

        self.W1 = 1e-2 * rd.randn(hidden_dim, input_dim)  # initialize weights
        self.W2 = 1e-2 * rd.randn(hidden_dim)

        self.actions = actions
        self.num_actions = len(actions)
        self.T = time_horizon
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.epsilon_param = epsilon_param
        self.lr = learning_rate
        self.num_gradient_steps = num_gradient_steps  # number of gradient steps we
        # take during each time period
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.leaky_coeff = leaky_coeff

        self.action_hist = np.zeros((self.T, input_dim))
        self.reward_hist = np.zeros(self.T)

    def _model_forward(self, input_actions):
        affine_out = np.sum(input_actions[:, np.newaxis, :] * self.W1, axis=2)
        relu_out = np.maximum(self.leaky_coeff * affine_out, affine_out)
        out = np.sum(relu_out * self.W2, axis=1)
        cache = (input_actions, affine_out, relu_out)
        return out, cache

    def _model_backward(self, out, cache, y):
        input_actions, affine_out, relu_out = cache
        dout = -(2 / self.noise_var) * (y - out)
        dW2 = np.sum(dout[:, np.newaxis] * relu_out, axis=0)
        drelu_out = dout[:, np.newaxis] * self.W2
        mask = (affine_out >= 0) + self.leaky_coeff * (affine_out < 0)
        daffine_out = mask * drelu_out
        dW1 = np.dot(daffine_out.T, input_actions)
        return dW1, dW2

    def _update_model(self, t):
        for i in range(self.num_gradient_steps):
            # sample minibatch
            batch_ind = rd.randint(t + 1, size=self.batch_size)
            action_batch = self.action_hist[batch_ind]
            reward_batch = self.reward_hist[batch_ind]

            out, cache = self._model_forward(action_batch)
            dW1, dW2 = self._model_backward(out, cache, reward_batch)
            dW1 /= self.batch_size
            dW2 /= self.batch_size
            dW1 += 2 / (self.prior_var * (t + 1)) * self.W1
            dW2 += 2 / (self.prior_var * (t + 1)) * self.W2

            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2

    def update_observation(self, observation, action, reward):
        t = observation
        self.action_hist[t] = self.actions[action]
        self.reward_hist[t] = reward
        self._update_model(t)
        self.lr *= self.lr_decay

    def pick_action(self, observation):
        u = rd.rand()
        if u < self.epsilon_param:
            action = rd.randint(self.num_actions)
        else:
            model_out, _ = self._model_forward(self.actions)
            action = np.argmax(model_out)
        return action


class TwoLayerNNEpsilonGreedyAnnealing(TwoLayerNNEpsilonGreedy):

    def pick_action(self, observation):
        t = observation
        epsilon = self.epsilon_param / (self.epsilon_param + t)
        u = rd.rand()
        if u < epsilon:
          action = rd.randint(self.num_actions)
        else:
          model_out, _ = self._model_forward(self.actions)
          action = np.argmax(model_out)
        return action


class TwoLayerNNDropout(TwoLayerNNEpsilonGreedy):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 actions,
                 time_horizon,
                 prior_var,
                 noise_var,
                 drop_prob=0.5,
                 learning_rate=1e-1,
                 num_gradient_steps=1,
                 batch_size=64,
                 lr_decay=1,
                 leaky_coeff=0.01):

        self.W1 = 1e-2 * rd.randn(hidden_dim, input_dim)
        self.W2 = 1e-2 * rd.randn(hidden_dim)

        self.actions = actions
        self.num_actions = len(actions)
        self.T = time_horizon
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.p = drop_prob
        self.lr = learning_rate
        self.num_gradient_steps = num_gradient_steps
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.leaky_coeff = leaky_coeff

        self.action_hist = np.zeros((self.T, input_dim))
        self.reward_hist = np.zeros(self.T)

    def _model_forward(self, input_actions):
        affine_out = np.sum(input_actions[:, np.newaxis, :] * self.W1, axis=2)
        relu_out = np.maximum(self.leaky_coeff * affine_out, affine_out)
        dropout_mask = rd.rand(*relu_out.shape) > self.p
        dropout_out = relu_out * dropout_mask
        out = np.sum(dropout_out * self.W2, axis=1)
        cache = (input_actions, affine_out, relu_out, dropout_mask, dropout_out)
        return out, cache

    def _model_backward(self, out, cache, y):
        input_actions, affine_out, relu_out, dropout_mask, dropout_out = cache
        dout = -(2 / self.noise_var) * (y - out)
        dW2 = np.sum(dout[:, np.newaxis] * relu_out, axis=0)
        ddropout_out = dout[:, np.newaxis] * self.W2
        drelu_out = ddropout_out * dropout_mask
        relu_mask = (affine_out >= 0) + self.leaky_coeff * (affine_out < 0)
        daffine_out = relu_mask * drelu_out
        dW1 = np.dot(daffine_out.T, input_actions)
        return dW1, dW2

    def pick_action(self, observation):
        model_out, _ = self._model_forward(self.actions)
        action = np.argmax(model_out)
        return action


class TwoLayerNNEnsembleSampling:

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 actions,
                 time_horizon,
                 prior_var,
                 noise_var,
                 num_models=10,
                 learning_rate=1e-3,
                 num_gradient_steps=1,
                 batch_size=64,
                 lr_decay=1,
                 leaky_coeff=0.01):

        self.M = num_models

        # initialize models by sampling perturbed prior means
        self.W1_model_prior = np.sqrt(prior_var) * rd.randn(self.M, hidden_dim,
                                                            input_dim)
        self.W2_model_prior = np.sqrt(prior_var) * rd.randn(self.M, hidden_dim)
        self.W1 = np.copy(self.W1_model_prior)
        self.W2 = np.copy(self.W2_model_prior)

        self.actions = actions
        self.num_actions = len(actions)
        self.T = time_horizon
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.lr = learning_rate
        self.num_gradient_steps = num_gradient_steps
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.leaky_coeff = leaky_coeff

        self.action_hist = np.zeros((self.T, input_dim))
        self.model_reward_hist = np.zeros((self.M, self.T))

    def _model_forward(self, m, input_actions):
        affine_out = np.sum(input_actions[:, np.newaxis, :] * self.W1[m], axis=2)
        relu_out = np.maximum(self.leaky_coeff * affine_out, affine_out)
        out = np.sum(relu_out * self.W2[m], axis=1)
        cache = (input_actions, affine_out, relu_out)
        return out, cache

    def _model_backward(self, m, out, cache, y):
        input_actions, affine_out, relu_out = cache
        dout = -(2 / self.noise_var) * (y - out)
        dW2 = np.sum(dout[:, np.newaxis] * relu_out, axis=0)
        drelu_out = dout[:, np.newaxis] * self.W2[m]
        mask = (affine_out >= 0) + self.leaky_coeff * (affine_out < 0)
        daffine_out = mask * drelu_out
        dW1 = np.dot(daffine_out.T, input_actions)
        return dW1, dW2

    def _update_model(self, m, t):
        for i in range(self.num_gradient_steps):
            # sample minibatch
            batch_ind = rd.randint(t + 1, size=self.batch_size)
            action_batch = self.action_hist[batch_ind]
            reward_batch = self.model_reward_hist[m][batch_ind]

            out, cache = self._model_forward(m, action_batch)
            dW1, dW2 = self._model_backward(m, out, cache, reward_batch)
            dW1 /= self.batch_size
            dW2 /= self.batch_size

            dW1 += 2 / (self.prior_var * (t + 1)) * (
                self.W1[m] - self.W1_model_prior[m])
            dW2 += 2 / (self.prior_var * (t + 1)) * (
                self.W2[m] - self.W2_model_prior[m])

            self.W1[m] -= self.lr * dW1
            self.W2[m] -= self.lr * dW2
        return

    def update_observation(self, observation, action, reward):
        t = observation
        self.action_hist[t] = self.actions[action]

        for m in range(self.M):
            m_noise = np.sqrt(self.noise_var) * rd.randn()
            self.model_reward_hist[m, t] = reward + m_noise
            self._update_model(m, t)

        self.lr *= self.lr_decay

    def pick_action(self, observation):
        m = rd.randint(self.M)
        model_out, _ = self._model_forward(m, self.actions)
        action = np.argmax(model_out)
        return action
