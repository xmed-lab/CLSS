
import sys
import math
import os
import time
import shutil
import datetime
import pandas as pd

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import subprocess



import models
import datasets
import utils
# from models.rnet2dp1 import r2plus1d_18_ctrst, SupConLoss_admargin

from models.ulb_rank_ssl import ulb_rank_cps, ulb_rank_prdlb_cps

from torch.distributions.normal import Normal

# from OrdinalEntropy import ordinal_entropy

@click.command("vol_reg_cmpl")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/wdaiaj/projects/datasources/AgeDB_DIR")
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="age")
@click.option("--model_name", type=click.Choice(['mc3_18', 'r2plus1d_18', 'r3d_18', 'r2plus1d_18_ncor']),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=30)   #### 30 is enough for full set, best usually not over 30
@click.option("--lr", type=float, default=0.0005)
@click.option("--weight_decay", type=float, default=1e-3)
@click.option("--lr_step_period", type=int, default=10)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--batch_size_two", type=int, default=8)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

@click.option("--full_test/--quick_test", default=False)
@click.option("--val_samp", type=int, default=1)
@click.option("--reduced_set/--full_set", default=True)
@click.option("--rd_label", type=int, default=2500)    #10518
@click.option("--rd_unlabel", type=int, default=5000)


@click.option("--ssl_mult", type=int, default=-1)
@click.option("--w_cps", type=float, default=1)

@click.option("--pad_param", type=int, default=12)

@click.option("--y_mean", type=float, default=43)
@click.option("--y_std", type=float, default=15)

@click.option("--lambda_val", type=float, default=2)

@click.option("--w_ctr", type=float, default=1)
@click.option("--w_ctrulb_0", type=float, default=1)
@click.option("--w_ctrulb_1", type=float, default=1)


@click.option("--excsplt/--allsplt", default=True)

def run(
    data_dir="/home/wdaiaj/projects/datasources/AgeDB_DIR",
    output=None,
    task="age",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=30,
    lr=0.0005,
    weight_decay=1e-3,
    lr_step_period=10,
    frames=32,
    period=2,
    num_workers=4,
    batch_size=32,
    batch_size_two=8,
    device=None,
    seed=0,

    full_test = False,
    val_samp = 1,
    reduced_set = True,
    rd_label = 610,
    rd_unlabel = 0,

    ssl_mult = -1,
    w_cps = 1,

    pad_param = 12,
    y_mean = 43,
    y_std = 15,
    lambda_val = 2,
    w_ctr = 1,
    w_ctrulb_0 = 1,
    w_ctrulb_1 = 1,
    excsplt = True
):
    

    command_args = sys.argv[:]

    print("Run with options:")
    for carg_itr in command_args:
        print(carg_itr)

    if reduced_set:
        if not os.path.isfile(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))):
            print("Generating new file list for ssl dataset")
            np.random.seed(0)
            
            data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
            data["SPLIT"].map(lambda x: x.upper())

            file_name_list = np.array(data[data['SPLIT']== 'TRAIN']['FileName'])
            np.random.shuffle(file_name_list)

            label_list = file_name_list[:rd_label]
            unlabel_list = file_name_list[rd_label:rd_label + rd_unlabel]

            data['SSL_SPLIT'] = "EXCLUDE"
            data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"
            data.loc[data['FileName'].isin(unlabel_list), 'SSL_SPLIT'] = "UNLABELED"

            data.to_csv(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel)),index = False)

    if ssl_mult != -1:
        ssl_mult_choice = ssl_mult
    else:
        ssl_mult_choice = rd_unlabel//rd_label

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):                            
        # print("worker id is", torch.utils.data.get_worker_info().id)
        # https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Set default output directory
    if output is None:
        assert 1==2, "need output option"

    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.isdir(os.path.join(output, "echonet_{}".format(bkup_tmstmp))):
        shutil.rmtree(os.path.join(output, "echonet_{}".format(bkup_tmstmp)))


    shutil.copytree("datasets", os.path.join(output, "echonet_{}".format(bkup_tmstmp), "datasets"))
    shutil.copytree("models", os.path.join(output, "echonet_{}".format(bkup_tmstmp), "models"))
    shutil.copy("vol_reg_oedNP_semi_ulbsep_ranksim.py", os.path.join(output, "echonet_{}".format(bkup_tmstmp), "vol_reg_oedNP_semi_ulbsep_ranksim.py"))

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        assert 1==2, "wrong parameter for device"


    if device.type == "cuda":
        # model = models.rnet2dp1.r2plus1d_18(pretrained=pretrained)
        # model = models.resnet50_CTR(pretrained=pretrained)
        model = models.resnet50_CTRNP(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        # model.fc.bias.data[0] = 43.4
        print("model.fc.bias.data[0]", model.fc.bias.data[0])
        model = torch.nn.DataParallel(model)

        
        model_1 = models.resnet50_CTRNP(pretrained=pretrained)
        model_1.fc = torch.nn.Linear(model_1.fc.in_features, 1)
        # model.fc.bias.data[0] = 43.4
        print("model.fc.bias.data[0]", model_1.fc.bias.data[0])
        model_1 = torch.nn.DataParallel(model_1)

    else:
        assert False, "wtf"
    model.to(device)
    model_1.to(device)


    if weights is not None:
        checkpoint = torch.load(weights)
        if checkpoint.get('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        elif checkpoint.get('state_dict_0'):
            model.load_state_dict(checkpoint['state_dict_0'])
        else:
            assert 1==2, "state dict not found"

    # Set up optimizer
    # optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("optim", optim)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, lr_step_period)


    mean, std = utils.get_mean_and_std(datasets.brt.Echo(root=data_dir, split="train")) # We can use whole dataset to get mean std, because not label dependent
    print("mean std", mean, std)


    kwargs = {"target_type": ['ground_truth'],
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset_trainsub = {}
    if reduced_set:
        # dataset_trainsub['lb'] = datasets.brt.Echo(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_mult = ssl_mult_choice, exclude_split = excsplt)
        dataset_trainsub['lb'] = datasets.brt.Echo(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_mult = ssl_mult_choice, exclude_split = excsplt)
        dataset_trainsub['unlb_0'] = datasets.brt.Echo(root=data_dir, split="train", **kwargs, pad=pad_param, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2, exclude_split = excsplt)
        # dataset_trainsub['unlb_1'] = datasets.Echo(root=data_dir, split="train", **kwargs, pad=12, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2)
    else:
        assert 1==2, "not possible"

    dataset['train'] = dataset_trainsub
    dataset["val"] = datasets.brt.Echo(root=data_dir, split="val", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))

    assert dataset_trainsub['lb'].get_labeled_length() == dataset_trainsub['unlb_0'].get_labeled_length(), "something wrong with contrast norm"
    assert dataset_trainsub['lb'].get_labeled_length() == dataset["val"].get_labeled_length(), "something wrong with contrast norm"
    criterion_cntrst_semi = models.ordinal_entropy
    criterion_cntrst = models.ordinal_entropy

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:

        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'], strict = False)
            optim_1.load_state_dict(checkpoint['opt_dict_1'])
            scheduler_1.load_state_dict(checkpoint['scheduler_dict_1'])

            np_rndstate_chkpt = checkpoint['np_rndstate']
            trch_rndstate_chkpt = checkpoint['trch_rndstate']

            np.random.set_state(np_rndstate_chkpt)
            torch.set_rng_state(trch_rndstate_chkpt)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")


        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:

                start_time = time.time()

                if device.type == "cuda":
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                
                ds = dataset[phase]
                if phase == "train":
                    dataloader_lb = torch.utils.data.DataLoader(
                        ds['lb'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    dataloader_unlb_0 = torch.utils.data.DataLoader(
                        ds['unlb_0'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    # dataloader_unlb_1 = torch.utils.data.DataLoader(
                    #     ds['unlb_1'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)



                    loss_tr, loss_reg_0, loss_cps,  ctr_ls, ctr_ulbls, yhat_0,yhat_1, y = run_epoch(model = [model, model_1], 
                                                                                    dataloader_lb = dataloader_lb, 
                                                                                    dataloader_unlb_0 = dataloader_unlb_0, 
                                                                                    train = phase == "train", 
                                                                                    optim = [optim, optim_1], 
                                                                                    device = device, 
                                                                                    w_cps = w_cps, 
                                                                                    y_mean = y_mean, 
                                                                                    y_std = y_std, 
                                                                                    w_ctr = w_ctr,
                                                                                    criterion_cntrst = criterion_cntrst_semi,
                                                                                    data_len = ds['lb'].get_labeled_length(),
                                                                                    run_dir = output, 
                                                                                    epoch = epoch,
                                                                                    lambda_val = lambda_val,
                                                                                    w_ctrulb_0 = w_ctrulb_0,
                                                                                    w_ctrulb_1 = w_ctrulb_1,
                                                                                    criterion_cntrst_val = criterion_cntrst,
                                                                                    batch_size_two = batch_size_two
                                                                                    )

                    r2_value_0 = sklearn.metrics.r2_score(y, yhat_0)
                    r2_value_1 = sklearn.metrics.r2_score(y, yhat_1)

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss_tr,
                                                                r2_value_0,
                                                                r2_value_1,
                                                                ctr_ls,
                                                                ctr_ulbls,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                loss_reg_0,
                                                                loss_cps))
                    f.flush()
                

                    # print("successful run until exit")
                    # exit()
                
                else:
                    ### for validation 
                    ### store seeds 
                    np_rndstate = np.random.get_state()
                    trch_rndstate = torch.get_rng_state()

                    r2_track = []
                    loss_track = []

                    for val_samp_itr in range(val_samp):
                        
                        print("running validation batch for seed =", val_samp_itr)

                        np.random.seed(val_samp_itr)
                        torch.manual_seed(val_samp_itr)
    
                        ds = dataset[phase]
                        dataloader = torch.utils.data.DataLoader(
                            ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))                        
                        
                        loss_valit, loss_ctr, yhat, y = run_epoch_val_avg(model = [model, model_1], 
                                                                            dataloader = dataloader, 
                                                                            train = False, 
                                                                            optim = None, 
                                                                            device = device, 
                                                                            save_all=False, 
                                                                            block_size=None, 
                                                                            y_mean = y_mean, 
                                                                            y_std = y_std, 
                                                                            data_len = ds.get_labeled_length(),
                                                                            criterion_cntrst = criterion_cntrst, 
                                                                            epoch = epoch,
                                                                            run_dir = output,
                                                                            lambda_val = lambda_val)

                        r2_track.append(sklearn.metrics.r2_score(y, yhat))
                        loss_track.append(loss_valit)

                    r2_value = np.average(np.array(r2_track))
                    loss = np.average(np.array(loss_track))

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                loss_ctr,
                                                                r2_value,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                0,
                                                                0))
            
                    for trck_write in range(len(r2_track)):
                        f.write(",{}".format(r2_track[trck_write]))

                    for trck_write in range(len(loss_track)):
                        f.write(",{}".format(loss_track[trck_write]))


                    f.write("\n")
                    f.flush()

                    np.random.set_state(np_rndstate)
                    torch.set_rng_state(trch_rndstate)

            
            scheduler.step()
            scheduler_1.step()

            best_model_loss = loss_valit

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_1': model_1.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                "best_model_loss": best_model_loss,
                'r2': r2_value,
                'opt_dict': optim.state_dict(),
                'opt_dict_1': optim_1.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'scheduler_dict_1': scheduler_1.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            # #### save based on total loss
            # if loss < bestLoss:
            #     print("saved best because {} < {}".format(loss, bestLoss))
            #     torch.save(save, os.path.join(output, "best.pt"))
            #     bestLoss = loss

            #### save based on reg loss
            # if loss < bestLoss:
            #     print("saved best because {} < {}".format(loss, bestLoss))
            #     torch.save(save, os.path.join(output, "best.pt"))
            #     bestLoss = loss
            
            if best_model_loss < bestLoss:
                print("saved best because {} < {}".format(best_model_loss, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = best_model_loss


        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            # checkpoint = torch.load(os.path.join(output, "best27_20.pt"))
            # checkpoint = torch.load(os.path.join(output, "best27_26.pt"))
            # checkpoint = torch.load(os.path.join(output, "best27_36.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)  
            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:
            split_list = ["test", "val"]
            for split in split_list: ### for CAMUS
                # Performance without test-time augmentation

                if not full_test:

                    for seed_itr in range(1):
                        np.random.seed(seed_itr)
                        torch.manual_seed(seed_itr)
                        
                        # uncomment this part up till continue if want to generate features
                        ds = datasets.brt.Echo(root=data_dir, split=split, **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
                        dataloader = torch.utils.data.DataLoader(
                            ds,
                            batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn)
                        total_loss, _, yhat, y = run_epoch_val_avg(model = [model, model_1], 
                                                                    dataloader = dataloader, 
                                                                    train = False, 
                                                                    optim = None, 
                                                                    device = device, 
                                                                    save_all=False, 
                                                                    block_size=None, 
                                                                    y_mean = y_mean, 
                                                                    y_std = y_std, 
                                                                    data_len = ds.get_labeled_length(),
                                                                    criterion_cntrst = criterion_cntrst,
                                                                    lambda_val = lambda_val)

                        for write_loop in range(5):
                            f.write("Seed is {}\n".format(seed_itr))
                            f.write("{} - {} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                            f.write("{} - {} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                            f.write("{} - {} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *tuple(map(math.sqrt, utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                            f.flush()

                        # with open(os.path.join(output, "z_{}_{}_s{}_strtfrmchk.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, seed)), "a") as f_start_frame:
                        #     for frame_itr in start_frame_record:
                        #         f_start_frame.write("{}\n".format(frame_itr))
                        #     f_start_frame.flush()

                        # with open(os.path.join(output, "z_{}_{}_s{}_vidpthchk.csv".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, seed)), "a") as f_vidpath:
                        #     for vidpath_itr in vidpath_record:
                        #         f_vidpath.write("{}\n".format(vidpath_itr))
                        #     f_vidpath.flush()

                        # # f.write("{} (one clip) R2:   {:.3f} \n".format(split, sklearn.metrics.r2_score(y, yhat)))
                        # # f.write("{} (one clip) MAE:  {:.2f} \n".format(split, sklearn.metrics.mean_absolute_error(y, yhat)))
                        # # f.write("{} (one clip) RMSE: {:.2f} \n".format(split, math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
                        # # f.flush()

                        # continue

                else:
                    assert False,"Full test does not apply"







def run_epoch(model, 
            dataloader_lb, 
            dataloader_unlb_0, 
            train, 
            optim, 
            device, 
            run_dir = None, 
            w_cps = 1, 
            y_mean = 43, 
            y_std = 15, 
            w_ctr = 1,
            criterion_cntrst = None,
            data_len = 0,
            epoch = 0,
            lambda_val = 2,
            w_ctrulb_0 = 0,
            w_ctrulb_1 = 0,
            criterion_cntrst_val = None,
            batch_size_two = 0):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """
    # print(type(model.module))
    # print(isinstance(model.module, models.rnet2dp1.VideoResNet_ncor))
    # exit()
    model[0].train(train)
    model[1].train(train)

    total = 0  # total training loss
    total_reg = 0 
    total_ctr = 0
    total_cps = 0
    n = 0      # number of videos processed
    total_ctrulb_0 = 0
    total_ctrulb_1 = 0
    yhat_0 = []
    yhat_1 = []
    y = []


    torch.set_grad_enabled(train)

    # assert len(dataloader_lb) == len(dataloader_unlb_0), "label unlabel not balanced label:{}  unlabel:{}".format(len(dataloader_lb),len(dataloader_unlb_0))
    # assert len(dataloader_lb) == len(dataloader_unlb_1), "label unlabel not balanced label:{}  unlabel:{}".format(len(dataloader_lb),len(dataloader_unlb_0))

    total_itr_num = min(len(dataloader_lb), len(dataloader_unlb_0))

    dataloader_lb_itr = iter(dataloader_lb)
    dataloader_unlb_0_itr = iter(dataloader_unlb_0)
    # dataloader_unlb_1_itr = iter(dataloader_unlb_1)

    for train_iter in range(total_itr_num):

        (X0, X1, outcome, outcome_cls ) = next(dataloader_lb_itr)

        # X, outcome = target_val_lb

        y.append(outcome.detach().cpu().numpy())
        X0 = X0.to(device)
        X1 = X1.to(device)
        outcome = outcome.to(device)

        all_output_0, all_output_feat_0 = model[0](X0)        
        all_output_1, all_output_feat_1 = model[1](X1)

        # loss_ctr = criterion_cntrst(all_output_feat,  (outcome - y_mean) / y_std)
        loss_ctr_0 = criterion_cntrst(all_output_feat_0,  (outcome - y_mean) / y_std)
        loss_ctr_1 = criterion_cntrst(all_output_feat_1,  (outcome - y_mean) / y_std)
        loss_ctr = loss_ctr_0 + loss_ctr_1


        loss_reg_0 = torch.nn.functional.mse_loss(all_output_0.view(-1), (outcome - y_mean) / y_std )
        yhat_0.append(all_output_0.view(-1).to("cpu").detach().numpy() * y_std + y_mean)        

        loss_reg_1 = torch.nn.functional.mse_loss(all_output_1.view(-1), (outcome - y_mean) / y_std )
        yhat_1.append(all_output_1.view(-1).to("cpu").detach().numpy() * y_std + y_mean)        

        loss_reg = loss_reg_0 + loss_reg_1
     
        
        (X_ulb_0, X_ulb_1, outcome_ulb, outcome_ulb_CLS) = next(dataloader_unlb_0_itr)
        outcome_ulb = outcome_ulb.to(device)
        X_ulb_0 = X_ulb_0.to(device)
        X_ulb_1 = X_ulb_1.to(device)
        
        all_output_unlb_0_pred_0, all_output_unlb_0_feat_0 = model[0](X_ulb_0)
        all_output_unlb_0_pred_1, all_output_unlb_0_feat_1 = model[1](X_ulb_1)

        all_output_unlb_0_pred_0_psl = all_output_unlb_0_pred_0.clone().detach()
        all_output_unlb_0_pred_1_psl = all_output_unlb_0_pred_1.clone().detach()


        cps_loss_0 = torch.nn.functional.mse_loss(all_output_unlb_0_pred_0.view(-1),all_output_unlb_0_pred_1_psl.view(-1) )
        cps_loss_1 = torch.nn.functional.mse_loss(all_output_unlb_0_pred_1.view(-1),all_output_unlb_0_pred_0_psl.view(-1) )
        cps_loss = cps_loss_0 + cps_loss_1


        if w_ctrulb_0 > 0:
            # assert False, "Not Done yet"
            loss_ulb_0, ft_rank, samples = ulb_rank_cps(all_output_unlb_0_feat_0, all_output_unlb_0_feat_1, lambda_val, batch_size_two)
            # print(ft_rank)
            if w_ctrulb_1 > 0:
                loss_ulb_1_0 = ulb_rank_prdlb_cps(all_output_unlb_0_pred_0, lambda_val, pred_inp=ft_rank, samples = samples)
                loss_ulb_1_1 = ulb_rank_prdlb_cps(all_output_unlb_0_pred_1, lambda_val, pred_inp=ft_rank, samples = samples)
                loss_ulb_1 = loss_ulb_1_0 + loss_ulb_1_1
            else:
                loss_ulb_1 = loss_reg * 0
        else:
            loss_ulb_0 = loss_reg * 0
            loss_ulb_1 = loss_reg * 0





        loss = loss_reg + w_cps * cps_loss + w_ctr * loss_ctr + w_ctrulb_0 * loss_ulb_0 + w_ctrulb_1 * loss_ulb_1



        if train:
            optim[0].zero_grad()
            optim[1].zero_grad()
            loss.backward()
            optim[0].step()
            optim[1].step()

        total += loss.item() * outcome.size(0)
        total_reg += loss_reg.item() * outcome.size(0)

        total_ctr += loss_ctr.item() * outcome.size(0)
        total_cps += cps_loss.item() * outcome.size(0)
        total_ctrulb_0 += loss_ulb_0.item() * outcome.size(0)
        total_ctrulb_1 += loss_ulb_1.item() * outcome.size(0)


        n += outcome.size(0)
        if train_iter % 10 == 0:
            # break
            print("phase {} itr {}/{}: ls {:.2f}({:.2f}) rg0 {:.4f} ({:.2f})  cps {:.6f} ({:.6f}) ctr {:.6f} ({:.6f}) ctrulb {:.7f} ({:.2f})  ctrulbPSL {:.7f} ({:.2f})    ".format(train,
                train_iter, total_itr_num, 
                total / n, loss.item(), 
                total_reg/n, loss_reg_0.item(), 
                total_cps / n, cps_loss.item(),
                total_ctr / n, loss_ctr.item(),
                total_ctrulb_0 / n, loss_ulb_0.item(),
                total_ctrulb_1 / n, loss_ulb_1.item()), flush = True)
        # pbar.update()

    yhat_0 = np.concatenate(yhat_0)
    yhat_1 = np.concatenate(yhat_1)

    y = np.concatenate(y)

    return total / n, total_reg / n, total_cps / n, total_ctr / n, total_ctrulb_0 / n, yhat_0, yhat_1,  y






def run_epoch_val_avg(model, 
                        dataloader, 
                        train, 
                        optim, 
                        device, 
                        save_all=False, 
                        block_size=None,  
                        y_mean = 43.3, 
                        y_std = 36, 
                        data_len = 0, 
                        criterion_cntrst = None, 
                        epoch = 0, 
                        run_dir = None,
                        lambda_val = 2):

    # model.train(False)
    model[0].train(False)
    model[1].train(False)


    total_ctr = 0
    total = 0  # total training loss
    n = 0      # number of videos processed

    kendal_total = 0

    yhat = []
    y = []
    val_iter = 0

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, _, outcome, outcome_cls) in dataloader:

                # X, outcome = target_val

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)
                # outcome_cls = outcome_cls.to(device)

                all_ouput_0, all_feat = model[0](X)
                all_ouput_1, all_feat = model[1](X)
                all_ouput = (all_ouput_0 + all_ouput_1) / 2
                yhat.append((all_ouput).view(-1).to("cpu").detach().numpy() * y_std + y_mean)
                loss = torch.nn.functional.mse_loss((all_ouput ).view(-1), (outcome - y_mean) / y_std )

                total += loss.item() * X.size(0)

                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) ".format(total / n, loss.item()))
                pbar.update()

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)


    return total / n, total_ctr / n,  yhat, y






if __name__ == '__main__':
    run()