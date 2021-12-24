import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # df = pd.read_pickle('eval_6_vit_model.csv.pkl')
    df = pd.read_pickle('eval_10_vit_700ep_model.csv.pkl')
    DETLA_T = 0.1

    runs_list_d = df.T.to_dict().values()

    runs_trajs_l = []
    for i, run in enumerate(runs_list_d):
        print(i, "run['runtime']", run['runtime'])
        run_xs = [0]
        run_ys = [0]

        for angle, vel in zip(run['steering_angles'], run['velocities']):
            x = run_xs[-1] + math.sin(angle) * vel * DETLA_T
            y = run_ys[-1] + math.cos(angle) * vel * DETLA_T
            run_xs.append(x)
            run_ys.append(y)
        runs_trajs_l.append([run_xs, run_ys])

    fig, ax = plt.subplots()
    fig.suptitle('Model ViT trained over 700 epochs')
    for i, [run_xs, run_ys] in enumerate(runs_trajs_l):
        ax.scatter(run_xs, run_ys, c=np.random.rand(3,), label='run number '+str(i))
    ax.set_xlim([-1, 1])
    ax.set_ylabel('y position')
    ax.set_xlabel('x position')
    ax.set_title('Trajectory of the agent in an evaluation run')
    ax.legend()
    plt.show()


