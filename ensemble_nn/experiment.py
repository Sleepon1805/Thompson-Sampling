import pandas as pd


##############################################################################

class Experiment:
    def __init__(self, agent, environment, n_steps,
                 rec_freq=1):
        self.agent = agent
        self.environment = environment
        self.n_steps = n_steps

        self.results = []
        self.data_dict = {}
        self.rec_freq = rec_freq

    def run_step(self, t):
        # Evolve the bandit (potentially contextual) for one step and pick action
        observation = self.environment.get_observation()
        action = self.agent.pick_action(observation)

        # Compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        expected_reward = self.environment.get_expected_reward(action)
        reward = self.environment.get_stochastic_reward(action)

        # Update the agent using realized rewards + bandit learing
        self.agent.update_observation(observation, action, reward)

        # Log whatever we need for the plots we will want to use.
        instant_regret = optimal_reward - expected_reward

        if (t + 1) % self.rec_freq == 0:
            self.data_dict = {'t': (t + 1),
                              'instant_regret': instant_regret,
                              'action': action}
            self.results.append(self.data_dict)

    def run_experiment(self):
        for t in range(self.n_steps):
            self.run_step(t)

        self.results = pd.DataFrame(self.results)
