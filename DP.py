import numpy as np

import matplotlib.pyplot as plt
from matplotlib.table import Table

from GridWorld import GridWorld

reward_decay = 0.9
grid_world_h = 5
grid_world_w = 5


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add form
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # row label
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label, loc='right', edgecolor='none', facecolor='none')
    # Column Label
    for j, label in enumerate(range(len(image))):
        tb.add_cell(grid_world_h, j, width, height / 2, text=label, loc='center', edgecolor='none',
                    facecolor='none')
    ax.add_table(tb)


def dp():
    # Initial value function
    value = np.zeros((grid_world_w, grid_world_h))
    while True:
        new_value = np.zeros(value.shape)
        # Go through all the states
        for i in range(0, grid_world_h):
            for j in range(0, grid_world_w):
                if (i == 2 and j == 2) or (i == 3 and j == 2):
                    continue

                values = []
                # Go through all the actions
                for action in range(env.n_actions):
                    base_coords, base_index = env.set_current_state(i, j)
                    env.render()
                    # Execute the action, move to the following state, and get an immediate reward
                    _, next_state_index, reward, done = env.step(action)
                    env.render()
                    if next_state_index != 'terminal':
                        next_i, next_j = next_state_index
                    else:
                        next_i, next_j = 0, 0
                    # Record value function q(s,a) = r + γ*v(s')
                    values.append(reward + reward_decay * value[next_i, next_j])
                # According to the optimal equation of behrman, find the maximum value function, and update q(s,a)
                new_value[i, j] = np.max(values)
        # Iteration termination condition: error less than 1e-4
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.title('$v_{*}$')
            plt.show()
            plt.close()
            break
        value = new_value
        print(value)
    # end of game
    print('game over')
    env.destroy()


env = GridWorld(grid_world_h, grid_world_w)
value = np.zeros((grid_world_h, grid_world_w))

env.after(10000, dp)
env.mainloop()
