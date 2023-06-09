import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import math
import copy
from sklearn.metrics import confusion_matrix

from utils import (
    create_folder,
    require_grad,
    no_grad,
    save_images,
    save_model,
    save_state_dict,
    gradient_penalty,
    CrossEntropy,
    ROC_AUC,
    plot_loss,
    plot_d_loss_components,
    plot_g_loss_components,
    plot_pred,
    plot_cls,
    plot_AUC,
    plot_pred_val,
    plot_cls_val,
    plot_AUC_val,
    plot_ROC_curve,
    save_classification_performances,
    show_confusion_matrix
)

from Load_dataset import LoadDataset
from utils import seed_everything
seed_everything(0)

def train_fn(
    D : nn.Module,                                   # the discriminator network
    G : nn.Module,                                   # the generator network
    opt_critic : torch.optim,                        # the optimizer of the discriminator
    opt_gen : torch.optim,                           # the optimizer of the generator
    scaler_critic : torch.cuda.amp.GradScaler,       # used to scale the loss of the discriminator
    scaler_gen : torch.cuda.amp.GradScaler,          # used to scale the loss of the generator
    train_loader : torch.utils.data.DataLoader,      # the dataloader of the training set
    train_dataset : torch.utils.data.Dataset,        # the training set
    validate : bool,                                 # if True, perform validation during training
    epoch_checking : int,                            # number of epochs after which start monitoring the performance on the validation set
    val_loader : torch.utils.data.DataLoader,        # the dataloader of the validation set
    val_dataset : torch.utils.data.Dataset,          # the validation set
    testing : bool,                                  # if True, perform testing after training
    use_best_model : bool,                           # if True, compute the performance on the test set using the models 
                                                        # that reached the greater generalization performance on the validation set
    test_loader : torch.utils.data.DataLoader,       # the dataloader of the test set
    test_dataset : torch.utils.data.Dataset,         # the test set
    step : int,                                      # current step of the progressive growing process
    alpha : float,                                   # fade-in factor for the current iteration (in [0, 1])
    num_epochs : int,                                # number of epochs to train
    img_size : int,                                  # size of the images at the current step
    class_size : int,                                # number of classes
    ncritic : int,                                   # number of times the discriminator is updated per each generator update
    z_dim : int,                                     # dimension of the latent space of the generator
    L_GP : float,                                    # weight of the gradient penalty term in the discriminator loss function
    L_cls1 : float,                                  # first weight of the classification loss
    L_cls2 : float,                                  # second weight of the classification loss
    model_directory : str,                           # path where to save the trained models
    image_directory : str,                           # path where to save the images
    device : str,                                    # 'cpu' or 'cuda'
    gpus : list,                                     # list of the gpus used to train the models
    disp_every_n_epochs : int                        # number of epochs after which save the models and images during the training, to check the results
):
    
    gen_loss_list = []; dis_loss_list = []
    d_real_list = []; d_fake_list = []
    cls_real_list = []; cls_fake_list = []
    gp_list = []; reg_list = []
    AUC_loss_real_list = []; AUC_loss_fake_list = [] 
    d_val_list = []; cls_val_list = []; AUC_val_list = []
    best_valid_loss = 0; best_epoch = 0
    best_G = copy.deepcopy(G).to('cpu'); best_D = copy.deepcopy(D).to('cpu')
    best_G_sd = best_G.state_dict(); best_D_sd = best_D.state_dict()
    best_optG_sd = opt_gen.state_dict(); best_optD_sd = opt_critic.state_dict()
    best_scalerG_sd = scaler_gen.state_dict(); best_scalerD_sd = scaler_critic.state_dict()
    model_dir = os.path.join(model_directory,'PACGAN_{}x{}'.format(img_size,img_size)); create_folder(model_dir)

    if testing and not validate:
        if use_best_model:
            # Set the number of training epochs based on the epoch that reached the greater generalization performance 
            # on th evalidation set
            checkpoint_file = os.path.join(model_directory[:-5], 'PACGAN_{}x{}'.format(img_size,img_size), 'generator_best.pt')
            checkpoint = torch.load(checkpoint_file)
            best_epoch = checkpoint['epoch']
            num_epochs = best_epoch

    for epoch in range(num_epochs):
        print("\nEpoch [{}/{}]".format(epoch+1,num_epochs), flush=True)

        gen_loss_mean = 0; dis_loss_mean = 0
        d_loss_real_mean = 0; d_loss_fake_mean = 0
        cls_loss_real_mean = 0; cls_loss_fake_mean = 0
        gp_mean = 0; reg_mean = 0
        AUC_loss_real_mean = 0; AUC_loss_fake_mean = 0

        d_size = len(train_dataset)

        # TRAINING
        D.train(); G.train()
        require_grad(D); require_grad(G)
        cur_time = time.time()
        for batch_idx, (real, labels) in enumerate(train_loader):
            real = real.to(device); labels = labels.to(device)
            cur_batch_size = real.shape[0]
            if batch_idx==0:
                n_batch = np.ceil(d_size/cur_batch_size) # number of minibatch in 1 epoch

            # TRAIN CRITIC ------------------------------------------------------------------------------------
            for _ in range(ncritic):   
                opt_critic.zero_grad()   
                z = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                fake = G(z, labels, alpha, step)
                real_critic, real_cls = D(real, alpha, step)
                fake_critic, fake_cls = D(fake.detach(), alpha, step)
                gp = gradient_penalty(D, real, fake, alpha, step, device=device)
                loss_critic = (
                    -(torch.mean(real_critic) - torch.mean(fake_critic))
                    + L_cls1 * CrossEntropy(labels, real_cls, device) + L_cls2 * CrossEntropy(labels, fake_cls, device)
                    + L_GP * gp
                    + (0.001 * torch.mean(real_critic ** 2)) # to avoid the critic to go too far away from 0
                )

                # Update D               
                scaler_critic.scale(loss_critic).backward()
                scaler_critic.step(opt_critic)
                scaler_critic.update()
            # -------------------------------------------------------------------------------------------------

            # Average of the discriminator loss function and all its terms over the epoch
            dis_loss_mean += loss_critic.item()/n_batch
            d_loss_real_mean += (-torch.mean(real_critic)).item()/n_batch
            d_loss_fake_mean += torch.mean(fake_critic).item()/n_batch            
            cls_loss_real_mean += CrossEntropy(labels, real_cls, device).item()/n_batch
            cls_loss_fake_mean += CrossEntropy(labels, fake_cls, device).item()/n_batch
            gp_mean += gp.item()/n_batch
            reg_mean += (0.001 * torch.mean(real_critic ** 2)).item()/n_batch
            AUC_loss_real_mean += ROC_AUC(labels, real_cls, class_size, device, AUC_loss_real_list[-1:]).item()/n_batch
            AUC_loss_fake_mean += ROC_AUC(labels, fake_cls, class_size, device, AUC_loss_fake_list[-1:]).item()/n_batch

            # TRAIN GENERATOR ----------------------------------------------------------------------------------
            opt_gen.zero_grad()
            z = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = G(z, labels, alpha, step)
            gen_fake, gen_cls = D(fake, alpha, step)
            real_critic, real_cls = D(real, alpha, step)
            loss_gen = (
                -torch.mean(gen_fake) 
                + L_cls1 *CrossEntropy(labels, gen_cls, device) + L_cls2 *CrossEntropy(labels, real_cls, device)
            )   
            
            # Update G
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            # --------------------------------------------------------------------------------------------------

            # Average of g_loss over the epoch
            gen_loss_mean += loss_gen.item()/n_batch

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / ((num_epochs * 0.5) * len(train_dataset))
            alpha = min(alpha, 1)

        # List of losses of each epoch
        gen_loss_list.append(gen_loss_mean)
        dis_loss_list.append(dis_loss_mean)
        d_real_list.append(d_loss_real_mean)
        d_fake_list.append(d_loss_fake_mean)
        cls_real_list.append(cls_loss_real_mean)
        cls_fake_list.append(cls_loss_fake_mean)
        gp_list.append(gp_mean)
        reg_list.append(reg_mean)
        AUC_loss_real_list.append(AUC_loss_real_mean)
        AUC_loss_fake_list.append(AUC_loss_fake_mean)

        if validate:
            # VALIDATION --------------------------------------------------------------------------------------
            D.eval(); G.eval()
            G = no_grad(G); D = no_grad(D)
            d_real_mean = 0
            CE_real_loss_val = 0; CE_fake_loss_val = 0
            AUC_real_loss_val = 0; AUC_fake_loss_val = 0
            for batch_idx, (val_real, val_labels) in enumerate(val_loader):
                val_real = val_real.to(device);  val_labels = val_labels.to(device) 
                cur_batch_size = val_real.shape[0]
                if batch_idx==0:
                    n_batch = np.ceil(len(val_dataset)/cur_batch_size) # number of minibatch in 1 epoch
                with torch.no_grad():       
                    z = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)    
                    fake = G(z, val_labels, alpha, step)
                    _, fake_cls = D(fake.detach(), alpha, step)
                    real_pred, real_cls = D(val_real, alpha, step)

                # Average values on the validation set
                d_real_mean += torch.mean(real_pred).item()/n_batch
                CE_real_loss_val += CrossEntropy(val_labels, real_cls, device).item()/n_batch
                CE_fake_loss_val += CrossEntropy(val_labels, fake_cls, device).item()/n_batch
                AUC_real_loss_val += ROC_AUC(val_labels, real_cls, class_size, device).item()/n_batch
                AUC_fake_loss_val += ROC_AUC(val_labels, fake_cls, class_size, device).item()/n_batch

            d_val_list.append(d_real_mean)
            cls_val_list.append(CE_real_loss_val)
            AUC_val_list.append(AUC_real_loss_val)

            # Save the best model
            check_loss = AUC_real_loss_val + AUC_fake_loss_val

            # Start monitor the performance on validation set after epoch_checking
            if epoch > epoch_checking[step]:
                if check_loss > best_valid_loss:
                    best_valid_loss = check_loss
                    best_epoch = epoch
                    best_G_sd.update(G.state_dict())
                    best_optG_sd.update(opt_gen.state_dict())
                    best_scalerG_sd.update(scaler_gen.state_dict())
                    best_D_sd.update(D.state_dict()) 
                    best_optD_sd.update(opt_critic.state_dict())
                    best_scalerD_sd.update(scaler_critic.state_dict())

        # ---------- --------------------------------------------------------------------------------------

        cur_time = time.time() - cur_time
        time_estimation = (num_epochs-epoch)*(cur_time)/3600
        m, h = math.modf(time_estimation)
        print('Estimated {} hours and {} minutes remaining for the image size {}x{}'.format(int(h), int(m*60), img_size, img_size), flush=True)

        # Define the path where save the images and plot the loss functions
        image_dir = os.path.join(image_directory,'PACGAN_{}x{}'.format(img_size,img_size)); create_folder(image_dir)

        # Save some examples of synthetic images during the training
        if epoch % disp_every_n_epochs == 0 or epoch == (num_epochs-1):
            labels = torch.LongTensor(np.arange(class_size)).to(device) # [0, 1]
            z = torch.randn(class_size, z_dim, 1, 1).to(device)
            sample_images = G(z, labels, alpha, step) # [2, 1, 256, 256]
            save_images(sample_images, image_dir, epoch)

    # Save best models
    if validate:
        save_state_dict(best_G_sd, best_optG_sd, best_scalerG_sd, best_epoch, model_dir, 'generator_best.pt')
        save_state_dict(best_D_sd, best_optD_sd, best_scalerD_sd, best_epoch, model_dir, 'discriminator_best.pt')
    else:
        save_model(G, opt_gen, scaler_gen, epoch+1, model_dir, 'generator_best.pt', device, gpus)
        save_model(D, opt_critic, scaler_critic, epoch+1, model_dir, 'discriminator_best.pt', device, gpus)

    # Plot losses during training 
    plot_loss(dis_loss_list, gen_loss_list, best_epoch, image_dir)
    plot_g_loss_components(d_real_list, d_fake_list, cls_real_list, cls_fake_list, image_dir, L_cls1, L_cls2)
    plot_d_loss_components(d_real_list, d_fake_list, cls_real_list, cls_fake_list, gp_list, image_dir, L_cls1, L_cls2, L_GP)  
    plot_pred(d_real_list, d_fake_list, image_dir)
    plot_cls(cls_real_list, cls_fake_list, image_dir)
    plot_AUC(AUC_loss_real_list, AUC_loss_fake_list, image_dir)

    if validate:
        # Plot losses during validation
        plot_pred_val(d_val_list, image_dir)
        plot_cls_val(cls_val_list, image_dir)
        plot_AUC_val(AUC_val_list, image_dir)
    else:
        if testing:
            # TESTING ------------------------------------------------------------------------------------------------
            D.eval(); D = no_grad(D)            
            CE_real_loss_test_best = 0; AUC_real_loss_test_best = 0

            # Initialize the lists where to concatenate all predictions, to compute metrics at the end.
            # In the final computation, we'll ignore the first element that we used for initialization.
            real_cls_list = torch.zeros_like(real_cls)[:1]
            test_labels_list = torch.zeros_like(labels)[:1]

            for batch_idx, (test_images, test_labels) in enumerate(test_loader):
                test_images = test_images.to(device) ;  test_labels = test_labels.to(device) 
                cur_batch_size = test_images.shape[0]
                if batch_idx==0:
                    n_batch = np.ceil(len(test_dataset)/cur_batch_size) # number of minibatch in 1 epoch
                with torch.no_grad():
                    _, real_cls = D(test_images, alpha, step)

                CE_real_loss_test_best += CrossEntropy(test_labels, real_cls, device)/n_batch
                AUC_real_loss_test_best += ROC_AUC(test_labels, real_cls, class_size, device)/n_batch
                
                # Memorize all the labels and predictions, to plot the ROC curve
                test_labels_list = torch.cat((test_labels_list.to(device), test_labels.to(device)))
                real_cls_list = torch.cat((real_cls_list.to(device), real_cls.to(device)))
            test_y = test_labels_list[1:]
            test_y_pred = real_cls_list[1:]

            # Plot the ROC curve
            plot_ROC_curve(test_y, test_y_pred, class_size, image_dir)

            # Plot the confusion matrix
            class_names = ['CN', 'AD']
            test_y_pred = np.argsort(test_y_pred.cpu(), axis=1)[:,-1]
            conf_matrix = confusion_matrix(test_y.cpu(), test_y_pred, normalize='true')
            show_confusion_matrix(conf_matrix, class_names, image_dir)

    # --------------------------------------------------------------------------------------------------------

    if img_size==256:
        # Save the classification performance of the final model
        filename = 'Classification_performance.txt' if validate else 'Classification_performance_test.txt'
        f = open(os.path.join(image_directory, filename), 'w')
        save_classification_performances(f, cls_real_list, cls_fake_list, AUC_loss_real_list, AUC_loss_fake_list)
        if validate:
            f.write('\nIn the validation set:\n')
            f.write('CrossEntropy(label,real_cls)_val: {}\n'.format(cls_val_list[best_epoch]))
            f.write('AUC(label,real_cls)_val: {}\n'.format(AUC_val_list[best_epoch]))
        else:
            if testing:
                f.write('\nIn the test set:\n')
                f.write('CrossEntropy(label,real_cls)_test: {}\n'.format(CE_real_loss_test_best))
                f.write('AUC(label,real_cls)_test: {}\n'.format(AUC_real_loss_test_best))
        f.close()