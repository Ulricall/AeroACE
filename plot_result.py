import matplotlib.pyplot as plt
import numpy as np
import trajectory
import matplotlib
import argparse
import os
from scipy.spatial import KDTree

t = trajectory.sin_forward()
def load_data(Name):
    logs = []
    for i in range(0, 5):
        log = np.load('logs/'+t.name+'/'+Name+'_'+str(i)+'.npy', allow_pickle=True)
        logs.append(log)
    logs = np.array(logs)
    return np.sum(logs, axis=0) / 5

dt = 0.01

axis_range = {
    'hover': [-0.5, 0.5, -0.5, 0.5, -0.3, 0.3],
    'fig8': [-3, 3, -3, 3, -2, 2],
    'spiral': [-10, 0, -5, 5, 0, 20],
    'terrain': [0, 10, 0, 10, 4.9, 6.3],
    'zigzag': [0, 5, -2, 2, -0.5, 0.5],
    'sin': [-2, 2, 0, 4, 0, 10]
}

def get_ground_truth(t, len):
    seq_len = int(len/dt+1)
    gt = np.zeros((seq_len,3))
    for i in range(seq_len):
        pd, vd, ad = t(i*dt)
        gt[i, :] = pd
    return gt

def plot_3D_trace(name):
    gt = get_ground_truth(t, 20)[:1950]
    log = load_data(name)[:1950]
    gt_tree = KDTree(gt)
    error, _ = gt_tree.query(log)
    # error = np.sqrt(np.sum((log - gt)**2, axis=1))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.15)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], '--', color='k', label='Ground Truth')

    sc = ax.scatter(log[:, 0], log[:, 1], log[:, 2],
                    c=error,
                    cmap='rainbow',
                    norm=norm,
                    s=5,
                    label=name)
    
    ranges = axis_range[t.name]
    ax.set_xlim([ranges[0], ranges[1]])
    ax.set_ylim([ranges[2], ranges[3]])
    ax.set_zlim([ranges[4], ranges[5]])
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Position Error [m]')
    # ax.legend(loc='upper right')

    if not os.path.exists('./traces'):
        os.makedirs('./traces')
    plt.savefig(f"traces/{t.name}_{name}.eps", dpi=150, bbox_inches='tight')
    plt.close(fig)

def show_project(name):
    gt = get_ground_truth(t, 20)
    log = load_data(name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    L = gt.shape[0]
    color = np.zeros(L)
    # fig8
    for i in range(L):
        color[i] = (log[i,0]+log[i,1]-gt[i,0]-gt[i,1])**2/2 + (log[i,2]-gt[i,2])**2
        for j in range(L):
            if (log[i,0]+log[i,1]-gt[j,0]-gt[j,1])**2/2 + (log[i,2]-gt[j,2])**2 < color[i]:
                color[i] = (log[i,0]+log[i,1]-gt[j,0]-gt[j,1])**2/2 + (log[i,2]-gt[j,2])**2
    plt.scatter((gt[:,0]+gt[:,1])/np.sqrt(2), gt[:,2], s=1, c='k', label='ground truth')
    plt.scatter((log[:,0]+log[:,1])/np.sqrt(2), log[:,2], c=np.sqrt(color), cmap='rainbow', label=name, s=2, norm=norm)
    # spiral & hover
    # for i in range(L):
    #     color[i] = (log[i,0]-gt[i,0])**2 + (log[i,1]-gt[i,1])**2
    #     for j in range(L):
    #         if (log[i,0]-gt[j,0])**2 + (log[i,1]-gt[j,1])**2 < color[i]:
    #             color[i] = (log[i,0]-gt[j,0])**2 + (log[i,1]-gt[j,1])**2
    # plt.scatter(gt[:,0], gt[:,1], s=1, c='k', label='ground truth')
    # plt.scatter(log[:,0], log[:,1], c=np.sqrt(color), cmap='rainbow', label=name, s=1, norm=norm)
    plt.colorbar()
    plt.xlim((0, 14))
    plt.ylim((4.9, 6.3))
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend(loc='upper right')
    if not os.path.exists('./projections'):
        os.makedirs('./projections')
    plt.savefig("projections/project_"+t.name+'_'+name+".eps", dpi=150)
    # plt.show()
    plt.close()

parser = argparse.ArgumentParser()
if __name__=='__main__':
    parser.add_argument('--trace', type=str, default='hover')
    args = parser.parse_args()
    if (args.trace=='hover'):
        t = trajectory.hover()
    elif (args.trace=='fig8'):
        t = trajectory.fig8()
    elif (args.trace=='spiral'):
        t = trajectory.spiral_up()
    elif (args.trace=='sin'):
        t = trajectory.sin_forward()
    elif (args.trace=='zigzag'):
        t = trajectory.ZigZag()
    elif (args.trace=='terrain'):
        t = trajectory.TerrainFollowingPath(base_path_planner=trajectory.LinePathXY(),
                                     terrain_model=trajectory.RandomGaussianTerrain())
    else:
        raise NotImplementedError
    Models = ['PID', 'OoD-Control', 'OMAC(deep)', 'Neural-Fly', 'Transformer', 'AeroACE']
    # Models = ['AeroACE']
    for model in Models:
        show_project(model)
        plot_3D_trace(model)
