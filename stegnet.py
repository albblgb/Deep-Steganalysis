import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging
import numpy as np
import math
from torchvision.utils import save_image
from tqdm.contrib import tzip

from models.StegNet import Model, initWeights
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.dataset import get_train_loader, get_val_loader, get_test_loader
from utils.dirs import mkdirs
import config as c


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_save_dir = os.path.join('checkpoints', 'StegNet')
results_save_dir = os.path.join('results', 'StegNet')
mkdirs(model_save_dir); mkdirs(results_save_dir)

logger_name = c.mode
logger_info(logger_name, log_path=os.path.join(results_save_dir, logger_name +'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('mode: {:s}'.format(c.mode))
logger.info('model: StegNet')
logger.info('train data dir: {:s}'.format(c.train_data_dir))
logger.info('val data dir: {:s}'.format(c.val_data_dir))
logger.info('test data dir: {:s}'.format(c.test_data_dir))


model = Model().to(device)
model = model.apply(initWeights)

if c.mode == 'test':

    test_loader = get_test_loader(c.test_data_dir, c.test_batch_size)
    model.load_state_dict(torch.load(c.pre_trained_stegnet_path))

    model.eval()

    with torch.no_grad():
        val_accuracy = []
        metric_monitor = MetricMonitor(float_precision=4)
        stream = tqdm(test_loader)

        for batch_idx, (inputs, labels) in enumerate(stream):

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            ################## forward ####################
            outputs = model(inputs)
            
            ################### loss ######################
            prediction = outputs.data.max(1)[1]

            accuracy = (
                prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            )
            val_accuracy.append(accuracy.item())
            metric_monitor.update("ACC", accuracy)
            stream.set_description(
                "Testing.  {metric_monitor}".format(metric_monitor=metric_monitor)
            )  

        val_acc_avg = np.mean(np.array(val_accuracy))

        logger.info('Testing, AVG_ACC: {:.4f}'.format(val_acc_avg))


else: # c.mode == 'train' 

    train_loader = get_train_loader(c.train_data_dir, c.train_batch_size,)
    val_loader = get_val_loader(c.val_data_dir, c.val_batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.weight_decay_step, gamma=c.gamma)

    for epoch in range(c.epochs):
        epoch += 1
        loss_history=[]
        train_accuracy = []
        ###############################################################
        #                            train                            # 
        ###############################################################
        model.train()
        metric_monitor = MetricMonitor(float_precision=4)
        stream = tqdm(train_loader)

        for batch_idx, train_batch in enumerate(stream):
            
            inputs = torch.cat((train_batch["cover"], train_batch["stego"]), 0)
            labels = torch.cat(
                (train_batch["label"][0], train_batch["label"][1]), 0
            )
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            ################## forward ####################
            outputs = model(inputs)

            ################### loss ######################
            loss = loss_fn(outputs, labels)

            ################### backword ##################
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_history.append(loss.item())
            prediction = outputs.data.max(1)[1]
            accuracy = (
                prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
            )
            train_accuracy.append(accuracy.item())

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("ACC", accuracy)
            stream.set_description(
                "Epoch: {epoch}/{total_epochs}. Training.   {metric_monitor}".format(epoch=epoch, total_epochs=c.epochs, metric_monitor=metric_monitor)
            )

        scheduler.step()  
        train_losses_avg = np.mean(np.array(loss_history))
        train_acc_avg = np.mean(np.array(train_accuracy))  

        ###############################################################
        #                              val                            # 
        ###############################################################
        model.eval()
        if epoch % c.val_freq == 0:
            with torch.no_grad():
                loss_history=[]
                val_accuracy = []
                metric_monitor = MetricMonitor(float_precision=4)
                stream = tqdm(val_loader)

                for batch_idx, val_batch in enumerate(stream):
                    inputs = torch.cat((val_batch["cover"], val_batch["stego"]), 0)
                    labels = torch.cat(
                        (val_batch["label"][0], val_batch["label"][1]), 0
                    )

                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)

                    
                    ################## forward ####################
                    outputs = model(inputs)
                    
                    ################### loss ######################
                    loss = loss_fn(outputs, labels)

                    loss_history.append(loss.item())
                    prediction = outputs.data.max(1)[1]

                    accuracy = (
                        prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
                    )
                    val_accuracy.append(accuracy.item())

                    metric_monitor.update("Loss", loss.item())
                    metric_monitor.update("ACC", accuracy)
                    stream.set_description(
                        "Epoch: {epoch}/{total_epochs}. Validating.  {metric_monitor}".format(epoch=epoch, total_epochs=c.epochs, metric_monitor=metric_monitor)
                    )  

                val_losses_avg = np.mean(np.array(loss_history))
                val_acc_avg = np.mean(np.array(val_accuracy))

                lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info('Epoch: {}/{}, Learning Rate: {:.5f} | Training, AVG_Loss {:.4f}, AVG_ACC: {:.4f} | Validating, AVG_Loss {:.4f}, AVG_ACC: {:.4f}'.format(epoch, c.epochs, lr, train_losses_avg, train_acc_avg, val_losses_avg, val_acc_avg))
        if epoch % c.save_freq == 0 and epoch >= c.strat_save_epoch:
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'checkpoint_%.3i' % epoch + '.pt'))
            
        


