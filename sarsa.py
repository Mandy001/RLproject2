import numpy as np
import pandas as pd

from GridWorld import GridWorld


class Sarsa(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if next_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)  # update


def update():
    for episode in range(100):
        # initial state
        _, state = env.reset()

        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            env.render()

            # RL take action and get next state and reward
            _, next_state_index, reward, done = env.step(action)

            # RL choose action based on next state
            next_action = RL.choose_action(str(next_state_index))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(state), action, reward, str(next_state_index), next_action)

            # swap state and action
            state = next_state_index
            action = next_action

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = GridWorld()
    RL = Sarsa(actions=list(range(env.n_actions)))

    env.after(10000, update)
    env.mainloop()
    print(RL.q_table)
