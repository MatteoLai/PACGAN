import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (RocCurveDisplay, roc_auc_score)
import copy
import sys

from Load_dataset import LoadDataset

def catch(data_train, labels_train, start_train_at_img_size, batch_sizes, critic_iterations, prog_epochs, class_size):
    if not(os.path.isdir(data_train)):
        print('Error: the folder DATA_TRAIN_DIRECTORY does not exist!', flush=True)
        sys.exit()
    if not (os.path.isfile(labels_train)):
        print('Error: the file LABEL_TRAIN_DIRECTORY does not exist!', flush=True)
        sys.exit()
    if not(start_train_at_img_size in [4, 8, 16, 32, 64, 128, 256]):
        print('Error: START_TRAIN_AT_IMG_SIZE should be one of the following value: 4, 8, 16, 32, 64, 128, 256', flush=True)
        sys.exit()
    if len(batch_sizes) != 7:
        print('Error: you should set BATCH_SIZES for each resolution! [4, 8, 16, 32, 64, 128, 256]', flush=True)
        sys.exit()
    if len(critic_iterations) != 7:
        print('Error: you should set CRITIC_ITERATIONS for each resolution! [4, 8, 16, 32, 64, 128, 256]', flush=True)
        sys.exit()
    if len(prog_epochs) != 7:
        print('Error: you should set BATCH_SIZES for each resolution! [4, 8, 16, 32, 64, 128, 256]', flush=True)
        sys.exit()
    if class_size<2:
        print('Error: CLASS_SIZE should be greater than 1!', flush=True)
        sys.exit()

def seed_everything(seed=42):
    '''Set the random seeds for riproducibility. 
    Note that reproducibility is not ensured on multi-GPU training
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_folder(directory):
    '''
    Check if the folder in "directory" already exist. If not, create it
    Args:
    directory (string): path to the folder to create
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print('Created folder: ', directory, flush=True)

def no_grad(model):
    """
    Args:
    model (nn.Module): model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    return model

def require_grad(model):
    """
    Args:
    model (nn.Module): model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    return model

def save_model(
    model: nn.Module,                               # model to save
    optimizer: torch.optim,                         # optimizer to save
    scaler: torch.cuda.amp.GradScaler,              # scaler to save
    epoch: int,                                     # current epoch
    model_dir: str,                                 # directory where to save the model
    name_file: str,                                 # name of the file of the model to save
    device: str,                                    # device where to save the model
    gpus: list                                      # list of the GPUs used during the training
):
    print("=> Saving model checkpoint", flush=True)
    # Save the model in cpu, to allow to upload it in every device
    model = model.to('cpu')
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                }, os.path.join(model_dir, name_file))
    # Move it back to config.DEVICE to go on with the training process
    model = model.to(device)
    if len(gpus)>1:
        model = nn.DataParallel(model, device_ids = gpus)

def save_state_dict(
    model_state_dict,                               # state dict of the model
    optimizer_state_dict,                           # state dict of the optimizer
    scaler_state_dict,                              # state dict of the scaler
    epoch: int,                                     # current epoch
    model_dir: str,                                 # directory where the model is saved
    name_file: str                                  # name of the file where the model is saved
):
    print("=> Saving best model (epoch: {})".format(epoch), flush=True)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'scaler_state_dict': scaler_state_dict,
                }, os.path.join(model_dir, name_file))

def load_model(
    model: nn.Module,                               # model to restore
    optimizer: torch.optim,                         # optimizer to restore
    scaler: torch.cuda.amp.GradScaler,              # scaler to restore
    img_size: int,                                  # image size
    model_directory: str,                           # directory where the model is saved
    name_file: str,                                 # name of the file where the model is saved
    device: str,                                    # device where the model is saved
    n_devices: int                                  # number of GPUs used during the training
):
    print("=> Loading model checkpoint", flush=True)
    checkpoint_file = os.path.join(model_directory, 'PACGAN_{}x{}'.format(img_size,img_size), name_file)

    checkpoint = torch.load(checkpoint_file)
    if n_devices>1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    model.to(device)

def restore_model(
    model: nn.Module,                               # model to restore
    optimizer: torch.optim,                         # optimizer to restore
    scaler: torch.cuda.amp.GradScaler,              # scaler to restore
    lr: float,                                      # learning rate
    img_size: int,                                  # image size
    model_directory: str,                           # directory where the model is saved
    name_file: str,                                 # name of the file where the model is saved
    device: str,                                    # device where the model is saved
    gpus: list                                      # list of the GPUs used during the training
):
    print("=> Loading model checkpoint", flush=True)
    checkpoint_file = os.path.join(model_directory, 'PACGAN_{}x{}'.format(img_size,img_size), name_file)
    checkpoint = torch.load(checkpoint_file)

    # Re-initialize model, optimizer and scaler
    n_devices = len(gpus)
    if n_devices>1:
        model = copy.deepcopy(model.module.to(device))
    else:
        model = copy.deepcopy(model.to(device))    
    optimizer = type(optimizer)(model.parameters(), lr=lr, betas=(0, 0.99))
    scaler = torch.cuda.amp.GradScaler()

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    model.to(device)

    if n_devices>1:
        model = nn.DataParallel(model, device_ids = gpus)

def get_loader(
    step: int,                                      # current step
    data_directory: str,                            # path to the nifti file
    labels_directory: str,                          # path to the CSV file with the labels
    batch_size: int,                                # batch size for the data loader
    n_devices: int = 1,                             # number of GPUs used during training
    selected_idxs: list = []                        # list of images and labels to extract from directories
):
    dataset = LoadDataset(batch_size, step, data_directory, labels_directory, n_devices, selected_idxs)

    g = torch.Generator()
    g.manual_seed(0)

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 0,
        pin_memory = True,
        shuffle = True,
        worker_init_fn = seed_worker,
        generator = g
    )
    return loader, dataset

def save_images(
    sample_images: torch.Tensor,                    # Generated images (one for each label)
    saved_image_directory: str,                     # Directory to save subplot of generated images
    epoch: int                                      # Current epoch
):
    """
    Save images generated at the current epoch
    """
    plt.close('all')
    image_grid = torchvision.utils.make_grid(sample_images.cpu().detach(), nrow=5, normalize=True)
    _, plot = plt.subplots(figsize=(12, 12))
    plt.axis('off')
    plot.imshow(image_grid.permute(1, 2, 0))
    if not os.path.isdir(saved_image_directory):
        os.makedirs(saved_image_directory)
        print('Created folder: ', saved_image_directory, flush=True)
    plt.savefig(os.path.join(saved_image_directory, 'epoch_') + '{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')

def gradient_penalty(
    critic: nn.Module,                              # the critic network
    real: torch.Tensor,                             # a batch of real images
    fake: torch.Tensor,                             # a batch of fake images
    alpha: float,                                   # fade-in factor for the current iteration (in [0, 1])
    train_step: int,                                # current training step
    device: str = "cpu"                             # 'cpu' or 'cuda'
):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores, _ = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def CrossEntropy(
    labels: torch.Tensor,                           # a batch of labels
    cls: torch.Tensor,                              # a batch of predictions performed by the discriminator
    device: str = 'cpu'                             # 'cpu' or 'cuda'
):
    loss = nn.CrossEntropyLoss().to(device)
    return loss(cls, labels.long())

def ROC_AUC(
    labels: torch.Tensor,                           # a batch of labels
    cls: torch.Tensor,                              # a batch of predictions performed by the discriminator
    class_size: int,                                # number of classes
    device: str,                                    # 'cpu' or 'cuda'
    precedent: list = []                            # list of the precedent values of AUC loss
):
    if sum(labels) == len(labels) or sum(labels) == 0:
        # In case the vectors "label" contains all the same elements ([0 0 ... 0] or [1 1 ... 1]),
        # roc_auc_score(labels,cls) returns error.
        # So, in case label is [0 0 ... 0] or [1 1 ... 1], we return the precedent value of AUC loss,
        # just to avoid the code to interrupt and obtain a reasonable loss plot at the end of the training.
        if len(precedent) == 0:
            # At the first iteration of the epoch the list is empty, so return 0 (just to avoid error).
            return  torch.tensor(0).to(device)
        else:
            return torch.tensor(precedent).to(device)
    labelsOH = F.one_hot(labels.long(), num_classes=class_size)
    labelsOH = labelsOH.float().cpu().detach().numpy()
    SoftMax = nn.Softmax(dim=1)
    cls_SM = SoftMax(cls)
    cls_SM = cls_SM.cpu().detach().numpy()
    AUC = 0
    for i in range(class_size):
        AUC_i = torch.tensor(roc_auc_score(labelsOH[:,i], cls_SM[:,i])).to(device)
        AUC += AUC_i
    return AUC/class_size

def plot_loss(
    dis_loss_list: list,                            # List of the Discriminator's loss at each epoch
    gen_loss_list: list,                            # List of the Generator's loss at each epoch
    best_epoch: int,                                # Epoch with the best performance
    saved_image_directory: str                      # Directory to save the plot
):
    plt.close('all')
    plt.plot(dis_loss_list, 'r-', label='Discriminator')
    plt.plot(gen_loss_list, 'b-', label='Generator')
    plt.plot(best_epoch, dis_loss_list[best_epoch], 'r*')
    plt.plot(best_epoch, gen_loss_list[best_epoch], 'r*')
    plt.legend(loc='best')
    plt.plot([0, len(dis_loss_list)-1], [0,0], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_losses.jpg'))

def plot_d_loss_components(
    d_real_list: list,                              # Mean of the Discriminator output to real images at each epoch
    d_fake_list: list,                              # Mean of the Discriminator output to fake images at each epoch
    cls_real_list: list,                            # Mean of the cross entropy between real labels and Discriminator class estimation for real images at each epoch
    cls_fake_list: list,                            # Mean of the cross entropy between real labels and Discriminator class estimation for fake images at each epoch
    gp_list: list,                                  # Mean of the Gradient Penalty computed at each epoch
    saved_image_directory: str,                     # Directory to save the plot
    L_cls1: int,                                    # Weight of the cross entropy between label and estimated class in the Discriminator loss
    L_cls2: int,                                    # Weight of the cross entropy between label and estimated class in the Discriminator loss
    L_GP: int                                       # Weight of the Gradient Penalty in the Discriminator loss
):
    plt.close('all')
    plt.plot(d_real_list, 'y-', label='-E[D(x)]')
    plt.plot(d_fake_list, 'm-', label='E[D(G(z))]')
    plt.plot(np.multiply(L_cls1,cls_real_list), 'g-', label='CrossEntropy(label, real_cls)*$\lambda_{cls1}$ ')
    plt.plot(np.multiply(L_cls2,cls_fake_list), 'c-',label='CrossEntropy(label, fake_cls)*$\lambda_{cls2}$')
    plt.plot(np.multiply(L_GP,gp_list), 'k-',label='GradientPenalty*$\lambda_{gp}$')
    plt.legend(loc='best')
    plt.plot([0, len(d_real_list)-1], [0,0], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator loss components')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_loss_components.jpg'))

def plot_g_loss_components(
    d_real_list: list,                              # Mean of the Discriminator output to real images at each epoch
    d_fake_list: list,                              # Mean of the Discriminator output to fake images at each epoch
    cls_real_list: list,                            # Mean of the cross entropy between real labels and Discriminator class estimation for real images at each epoch
    cls_fake_list: list,                            # Mean of the cross entropy between real labels and Discriminator class estimation for fake images at each epoch
    saved_image_directory: str,                     # Directory to save the plot
    L_cls1: int,                                    # Weight of the cross entropy between label and estimated class in the Generator loss
    L_cls2: int                                     # Weight of the cross entropy between label and estimated class in the Generator loss
):
    plt.close('all')
    plt.plot(np.multiply(-1,d_fake_list), 'm-', label='-E[D(G(z))]')
    plt.plot(np.multiply(L_cls2,cls_real_list), 'g-', label='CrossEntropy(label, real_cls)*$\lambda_{cls2}$ ')
    plt.plot(np.multiply(L_cls1,cls_fake_list), 'c-',label='CrossEntropy(label, fake_cls)*$\lambda_{cls1}$')
    plt.legend(loc='best')
    plt.plot([0, len(d_real_list)-1], [0,0], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator loss components')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_g_loss_components.jpg'))

def plot_pred(
    d_real_list: list,                  # Mean of the Discriminator output to real images at each epoch
    d_fake_list: list,                  # Mean of the Discriminator output to fake images at each epoch
    saved_image_directory: str          # Directory to save the plot
):
    plt.close('all')
    plt.plot(np.multiply(-1, d_real_list), 'y-', label='E[D(x)]')
    plt.plot(d_fake_list, 'm-', label='E[D(G(z))]')
    plt.legend(loc='best')
    plt.plot([0, len(d_real_list)-1], [0,0], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator prediction output')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_predictions.jpg'))


def plot_cls(
    cls_real_list: list,                  # Mean of the cross entropy between real labels and Discriminator class estimation for real images at each epoch
    cls_fake_list: list,                  # Mean of the cross entropy between real labels and Discriminator class estimation for fake images at each epoch
    saved_image_directory: str            # Directory to save the plot
):
    plt.close('all')
    plt.plot(cls_real_list, 'g-', label='CrossEntropy(label, real_cls)')
    plt.plot(cls_fake_list, 'c-', label='CrossEntropy(label, fake_cls)')
    plt.legend(loc='best')
    plt.plot([0, len(cls_real_list)-1], [0,0], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy(label, cls)')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_cls.jpg'))

def plot_AUC(
    AUC_loss_real_list: list,             # Mean of the ROC AUC between real labels and Discriminator class estimation for real images at each epoch
    AUC_loss_fake_list: list,             # Mean of the ROC AUC between real labels and Discriminator class estimation for fake images at each epoch
    saved_image_directory: str            # Directory to save the plot
):
    plt.close('all')
    plt.plot(AUC_loss_real_list, 'g-', label='AUC(label, real_cls)')
    plt.plot(AUC_loss_fake_list, 'c-', label='AUC(label, fake_cls)')
    plt.legend(loc='best')
    plt.plot([0, len(AUC_loss_fake_list)-1], [0,0], 'c--')
    plt.plot([0, len(AUC_loss_fake_list)-1], [0.5,0.5], 'r--')
    plt.plot([0, len(AUC_loss_fake_list)-1], [1,1], 'c--')
    plt.xlabel('Epochs')
    plt.ylabel('AUC(label, cls)')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_cls_AUC.jpg'))


def plot_pred_val(
    d_real_list: list,                # Mean of the Discriminator output to images of the validation set at each epoch
    saved_image_directory: str        # Directory to save the plot
):
    plt.close('all')
    plt.plot(d_real_list, 'y-')
    plt.plot([0, len(d_real_list)-1], [0,0], 'c--')
    plt.title('E[D(x)] validation set')
    plt.xlabel('Epochs')
    plt.ylabel('E[D(x)]')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_test_d.jpg'))

def plot_cls_val(
    cls_real_list: list,              # Mean of the cross entropy between real labels and Discriminator class estimation for images of the validation set at each epoch
    saved_image_directory: str        # Directory to save the plot
):
    plt.close('all')
    plt.plot(cls_real_list, 'g-')
    plt.plot([0, len(cls_real_list)-1], [0,0], 'c--')
    plt.title('CrossEntropy(label, real_cls) validation set')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy(label, real_cls)')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_test_cls.jpg'))

def plot_AUC_val(
    AUC_real_list: list,              # Mean of the ROC AUC between real labels and Discriminator class estimation for images of the validation set at each epoch
    saved_image_directory: str        # Directory to save the plot
):
    plt.close('all')
    plt.plot(AUC_real_list, 'g-')
    plt.plot([0, len(AUC_real_list)-1], [0,0], 'c--')
    plt.plot([0, len(AUC_real_list)-1], [0.5,0.5], 'r--')
    plt.plot([0, len(AUC_real_list)-1], [1,1], 'c--')
    plt.title('AUC(label, real_cls) validation set')
    plt.xlabel('Epochs')
    plt.ylabel('AUC(label, real_cls)')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_test_cls_AUC.jpg'))

def plot_ROC_curve(
    test_labels: torch.Tensor,            # Labels of the test set
    real_cls: torch.Tensor,               # Discriminator class estimation for images of the test set
    class_size: int,                      # Number of classes
    saved_image_directory: str            # Directory to save the plot
):
    labelsOH = F.one_hot(test_labels.long(), num_classes=class_size)
    labelsOH = labelsOH.float().cpu().detach().numpy()
    SoftMax = nn.Softmax(dim=1)
    cls_SM = SoftMax(real_cls)
    cls_SM = cls_SM.cpu().detach().numpy()
    n_iter = class_size if class_size!=2 else 1
    for class_id in range(n_iter):
        RocCurveDisplay.from_predictions(
        labelsOH[:, class_id],
        cls_SM[:, class_id],
        name="Class {} vs the rest".format(class_id) if class_size!=2 else "Class 0 vs class 1",
        color="blue",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.savefig(os.path.join(saved_image_directory, 'PACGAN_test_ROC{}.jpg'.format(class_id)))

def save_classification_performances(
    f,                                   # File where to write the classification performance
    cls_real_list: list,                 # Mean of the cross entropy between real labels and Discriminator class estimation for real images
    cls_fake_list: list,                 # Mean of the cross entropy between real labels and Discriminator class estimation for fake images
    AUC_loss_real_list: list,            # Mean of the ROC AUC between real labels and Discriminator class estimation for real images
    AUC_loss_fake_list: list             # Mean of the ROC AUC between real labels and Discriminator class estimation for fake images
):
    ''' 
    Write the txt file with the classification performance of the Discriminator with images of the
    training, validation, and (if TESTING=True) test set.
    '''
    f.write('Classification performance at the end of the training:\n')
    f.write('\nIn the training set:\n')
    f.write('CrossEntropy(label, real_cls): {}\n'.format(cls_real_list[-1]))
    f.write('CrossEntropy(label, fake_cls): {}\n'.format(cls_fake_list[-1]))
    f.write('AUC(label, real_cls): {}\n'.format(AUC_loss_real_list[-1]))
    f.write('AUC(label, fake_cls): {}\n'.format(AUC_loss_fake_list[-1]))

def show_confusion_matrix(
    conf_matrix: torch.Tensor,                # Confusion matrix
    class_names: list,                        # List of the classes
    saved_image_directory: str,               # Directory to save the confusion matrix plot
    figsize: tuple = (10,10)                  # Size of the plot
):
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.matshow(conf_matrix)
    tick_marks = np.arange(len(class_names))
    _=plt.xticks(tick_marks, class_names,rotation=45)
    _=plt.yticks(tick_marks, class_names)
    _=plt.ylabel('Real')
    _=plt.xlabel('Predicted')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, '{0:.1%}'.format(conf_matrix[i, j]),
                            ha='center', va='center', color='w')
    plt.savefig(os.path.join(saved_image_directory, 'PACGAN_test_ConfusionMatrix.jpg'))