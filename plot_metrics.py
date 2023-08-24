import matplotlib.pyplot as plt
import pandas as pd

path = "Path/to/Results.csv"
save_path = "Path/to_folder/where/to/save/plots"

df = pd.read_csv(path)

def plot_hyperparameters(data, x_label, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(title, fontsize = 20)

    if x_label=='z_dim':
        xaxis = ['100', '300', '512']
    elif x_label=='Emb_dim':
        xaxis = ['2', '3', '4', '5', '6', '7', '8']
    elif x_label=='lmb_cls':
        xaxis = ['2', '3', '4', '5', '6']

    p1, = ax.plot(xaxis, data['FID_rf'], marker='o', color='blue', label='FID')
    ax.yaxis.label.set_color(p1.get_color())
    ax.tick_params(axis='y', colors=p1.get_color())

    ax2 = ax.twinx()
    p2, = ax2.plot(xaxis, data['KID_rf'], marker='o', color='red', label='KID')
    ax2.yaxis.label.set_color(p2.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())

    ax3 = ax.twinx()
    bias=0.07
    ax3.spines['right'].set_position(('axes', 1 + bias))
    p3, = ax3.plot(xaxis, data['SSIM_rf'], marker='o', color='green', label='SSIM')
    ax3.yaxis.label.set_color(p3.get_color())
    ax3.tick_params(axis='y', colors=p3.get_color())

    ax.set_xlabel(x_label, fontsize = 14)

    fig.legend(bbox_to_anchor=(0.105, 1))
    plt.tight_layout()

    plt.savefig(save_path+'metrics_'+x_label+'.png')

def plot_classification(data, x_label, title):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize = 15)

    if x_label=='z_dim':
        xaxis = ['100', '300', '512']
    elif x_label=='Emb_dim':
        xaxis = ['2', '3', '4', '5', '6', '7', '8']
    elif x_label=='lmb_cls':
        xaxis = ['2', '3', '4', '5', '6']

    ax.plot(xaxis, data, marker='o', color='Green')
    ax.set_ylim(0.5, 1)
    ax.set_xlabel(x_label, fontsize = 12)
    ax.set_ylabel('$AUC_{val}$', fontsize = 12)
    plt.savefig(save_path+'classification_'+x_label+'.png')

means_z = pd.concat([df[df['z_dim']==100].mean(), df[df['z_dim']==300].mean(), df[df['z_dim']==512].mean()], axis=1)
plot_hyperparameters(means_z.transpose(), 'z_dim', 'Assessment metrics varying $z_{dim}$')

means_e = pd.concat([df[df['Emb_dim']==2].mean(), df[df['Emb_dim']==3].mean(), df[df['Emb_dim']==4].mean(), df[df['Emb_dim']==5].mean(), df[df['Emb_dim']==6].mean(), df[df['Emb_dim']==7].mean(), df[df['Emb_dim']==8].mean()], axis=1)
plot_hyperparameters(means_e.transpose(), 'Emb_dim', 'Assessment metrics varying $emb_{dim}$')

means_l = pd.concat([df[df['lmb_cls']==2].mean(), df[df['lmb_cls']==3].mean(), df[df['lmb_cls']==4].mean(), df[df['lmb_cls']==5].mean(), df[df['lmb_cls']==6].mean()], axis=1)
plot_hyperparameters(means_l.transpose(), 'lmb_cls', 'Assessment metrics varying $\lambda_{cls}$')

cls_z = [df[df['z_dim']==100].mean()['AUC_val'], df[df['z_dim']==300].mean()['AUC_val'], df[df['z_dim']==512].mean()['AUC_val']]
plot_classification(cls_z, 'z_dim', 'Classification performance varying $z_{dim}$')

cls_e = [df[df['Emb_dim']==2].mean()['AUC_val'], df[df['Emb_dim']==3].mean()['AUC_val'], df[df['Emb_dim']==4].mean()['AUC_val'], df[df['Emb_dim']==5].mean()['AUC_val'], df[df['Emb_dim']==6].mean()['AUC_val'], df[df['Emb_dim']==7].mean()['AUC_val'], df[df['Emb_dim']==8].mean()['AUC_val']]
plot_classification(cls_e, 'Emb_dim', 'Classification performance varying $emb_{dim}$')

cls_l = [df[df['lmb_cls']==2].mean()['AUC_val'], df[df['lmb_cls']==3].mean()['AUC_val'], df[df['lmb_cls']==4].mean()['AUC_val'], df[df['lmb_cls']==5].mean()['AUC_val'], df[df['lmb_cls']==6].mean()['AUC_val']]
plot_classification(cls_l, 'lmb_cls', 'Classification performance varying $\lambda_{cls}$')