"""
Class for training a fully supervised classifier
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy
import heapq

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData
from watchmal.engine.EarlyStopping import EarlyStopping

class ClassifierEngine:
    def __init__(self, model, rank, gpu, dump_path):
        """
        Args:
            model       ... model object that engine will use in training or evaluation
            rank        ... rank of process among all spawned processes (in multiprocessing mode)
            gpu         ... gpu that this process is running on
            dump_path   ... path to store outputs in
        """
        # create the directory for saving the log and dump files
        self.epoch = 0.
        self.best_validation_loss = 1.0e10
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)

        # TODO: print model summary
        #summary(self.model, (38, 29, 60))

        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}

        # define the placeholder attributes
        self.data = None
        self.labels = None
        self.loss = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # early stopper
        self.early_stop = False
    
    def configure_optimizers(self, optimizer_config):
        """
        Set up optimizers from optimizer config

        Args:
            optimizer_config    ... hydra config specifying optimizer object
        """
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders config

        Args:
            data_config     ... hydra config specifying dataset
            loaders_config  ... hydra config specifying dataloaders
            is_distributed  ... boolean indicating if running in multiprocessing mode
            seed            ... seed to use to initialize dataloaders
        
        Parameters:
            self should have dict attribute data_loaders
        """
        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)
    
    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations

        Args:
            metric_dict         ... dict containing values that are tensor outputs of a single process
        
        Returns:
            global_metric_dict  ... dict containing concatenated list of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, array in zip(metric_dict.keys(), metric_dict.values()):
            tensor = torch.as_tensor(array).to(self.device)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[name] = torch.cat(global_tensor)
        
        return global_metric_dict

    def forward(self, train=True, mc_dropout=False, fwd_passes=None):
        """
        Compute predictions and metrics for a batch of data

        Args:
            train   ... whether to compute gradients for backpropagation

        Parameters:
            self should have attributes data, labels, model, criterion, softmax
        
        Returns:
            dict containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU (if using CPU this has no effect)
            data = self.data.to(self.device)
            labels = self.labels.to(self.device)

            # Initialize results dict
            result = {}

            model_out, softmax = self.model(data)

            if mc_dropout:
                probs = torch.unsqueeze(softmax, 2)
                raw_output = torch.unsqueeze(model_out, 2)

                # TODO: odd number to avoid tie in mode
                for i in range(fwd_passes-1):
                    model_out, softmax = self.model(data)
                    probs = torch.cat((probs, torch.unsqueeze(softmax, 2)), 2)
                    raw_output = torch.cat((raw_output, torch.unsqueeze(model_out, 2)), 2)

                softmax = torch.mean(probs, 2)
                model_out = torch.mean(raw_output, 2)

                # Uncertainty measurements
                epsilon = sys.float_info.min
                # TODO: reduction only because for 2 classes both uncertainties are equal
                total_variance = torch.std(probs[:, 0], dim=1)
                entropy = - torch.sum(softmax * torch.log(softmax + epsilon), dim=1)
                mutual_info = entropy - torch.mean(torch.sum(-probs * torch.log(probs + epsilon), dim=1), dim=-1)
                # TODO: only implemented for 2 class (for more classes: heapq.nlargest(2, probs) and subtract)
                margin_confidence = torch.mean(torch.abs(probs[:, 0, :] - probs[:, 1, :]), dim=1)
                # TODO: avoid for loop
                freq_modes = torch.stack([torch.max(torch.unique(t, return_counts=True)[1]) for t in torch.unbind(torch.argmax(probs, dim=1))])
                variation_ratio = 1 - (freq_modes/fwd_passes)

                uncertainties = torch.stack((total_variance, entropy, mutual_info, margin_confidence, variation_ratio), dim=-1)
                result['uncertainty'] = uncertainties

            predicted_labels = torch.argmax(model_out, dim=-1)

            result['predicted_labels'] = predicted_labels
            result['softmax'] = softmax
            result['raw_pred_labels'] = model_out

            self.loss = self.criterion(model_out, labels)
            accuracy = (predicted_labels == labels).sum().item() / float(predicted_labels.nelement())

            result['loss'] = self.loss.item()
            result['accuracy'] = accuracy
        
        return result
    
    def backward(self):
        """
        Backward pass using the loss computed for a mini-batch

        Parameters:
            self should have attributes loss, optimizer
        """
        self.optimizer.zero_grad()  # reset accumulated gradient
        self.loss.backward()        # compute new gradient
        self.optimizer.step()       # step params
    
    # ========================================================================
    # Training and evaluation loops
    
    def train(self, train_config):
        """
        Train the model on the training set

        Args:
            train_config    ... config specigying training parameters
        
        Parameters:
            self should have attributes model, data_loaders
        
        Outputs:
            val_log      ... csv log containing iteration, epoch, loss, accuracy for each iteration on validation set
            train_logs   ... csv logs containing iteration, epoch, loss, accuracy for each iteration on training set
            
        Returns: None
        """
        # initialize training params
        epochs          = train_config.epochs
        report_interval = train_config.report_interval
        val_interval    = train_config.val_interval
        num_val_batches = train_config.num_val_batches
        checkpointing   = train_config.checkpointing

        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()

        # initialize epoch and iteration counters
        self.epoch = 0.
        self.iteration = 0

        # keep track of the validation loss
        self.best_validation_loss = 1.0e10

        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])

        # initialize the early_stopper object
        #early_stopper = EarlyStopping(**train_config.early_stopping)

        # global training loop for multiple epochs
        while (floor(self.epoch) < epochs):
            if self.rank == 0:
                print('\nEpoch', floor(self.epoch), 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))


            start_time = time()
            iteration_time = start_time

            train_loader = self.data_loaders["train"]

            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            # local training loop for batches in a single epoch
            for i, train_data in enumerate(self.data_loaders["train"]):
                
                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    self.validate(val_iter, num_val_batches, checkpointing, early_stopper=None)
                
                # Train on batch
                self.data = train_data['data']
                self.labels = train_data['labels']

                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.forward(True)

                #Call backward: backpropagate error and update weights using loss = self.loss
                self.backward()

                # update the epoch and iteration
                self.epoch += 1. / len(self.data_loaders["train"])
                self.iteration += 1
                
                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": res["loss"], "accuracy": res["accuracy"]}
                
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    print("Training: ... Iteration %d ... Epoch %1.2f ... Training Loss %1.3f ... Training Accuracy %1.3f ... Time Elapsed %1.3f ... Iteration Time %1.3f" %
                           (self.iteration, self.epoch, res["loss"], res["accuracy"], iteration_time - start_time, iteration_time - previous_iteration_time))

                if self.epoch >= epochs:
                    break

            if self.early_stop:
                break
        
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def validate(self, val_iter, num_val_batches, checkpointing, early_stopper):
        # set model to eval mode
        self.model.eval()
        val_metrics = {"iteration": self.iteration, "loss": 0., "accuracy": 0., "saved_best": 0}
        for val_batch in range(num_val_batches):
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                #print("Fetching new validation iterator...")
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)

            # extract the event data from the input data tuple
            self.data = val_data['data']
            self.labels = val_data['labels']

            val_res = self.forward(False)

            val_metrics["loss"] += val_res["loss"]
            val_metrics["accuracy"] += val_res["accuracy"]
        # return model to training mode
        self.model.train()
        # record the validation stats
        val_metrics["loss"] /= num_val_batches
        val_metrics["accuracy"] /= num_val_batches
        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "accuracy": np.array([val_metrics["accuracy"]])}

        if self.is_distributed:
            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                global_val_metrics[name] = np.array(tensor.cpu())
        else:
            global_val_metrics = local_val_metrics

        if self.rank == 0:
            # Save if this is the best model so far
            global_val_loss = np.mean(global_val_metrics["loss"])
            global_val_accuracy = np.mean(global_val_metrics["accuracy"])

            val_metrics["loss"] = global_val_loss
            val_metrics["accuracy"] = global_val_accuracy
            val_metrics["epoch"] = self.epoch

            print("Validation: ... Val Loss %1.3f ... Val Accuracy %1.3f" % (val_metrics["loss"], val_metrics["accuracy"]))

            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                print('Best validation loss so far!: {}'.format(self.best_validation_loss))
                self.save_state(best=True)
                val_metrics["saved_best"] = 1

            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state(best=False)

            # TODO: in distributed?
            #early_stopper(val_metrics["loss"])
            #self.early_stop = early_stopper.early_stop

            self.val_log.record(val_metrics)
            self.val_log.write()
            self.val_log.flush()

    def evaluate(self, test_config):
        """
        Evaluate the performance of the trained model on the test set

        Args:
            test_config ... hydra config specifying evaluation parameters
        
        Parameters:
            self should have attributes model, data_loaders, dirpath
        
        Outputs:
            indices     ... index in dataset of each event
            labels      ... actual label of each event
            predictions ... predicted label of each event
            softmax     ... softmax output over classes for each event
            
        Returns: None
        """
        print("evaluating in directory: ", self.dirpath)

        # MC dropout parameters
        self.mc_dropout = test_config.mc_dropout
        self.fwd_passes = test_config.fwd_passes

        # Variables to output at the end
        eval_loss = 0.0
        eval_acc = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()

            # Enable Dropout in evaluation
            if self.mc_dropout:
                print(f'\nEvaluation with MC Dropout, forward passes: {self.fwd_passes}')
                self.enable_dropout(self.model)

            # Variables for the confusion matrix
            loss, accuracy, indices, labels, predictions, softmaxes, uncertainties = [],[],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):
                
                # load data
                self.data = eval_data['data']
                self.labels = eval_data['labels']

                eval_indices = eval_data['indices']
                
                # Run the forward procedure and output the result
                result = self.forward(train=False, mc_dropout=self.mc_dropout, fwd_passes=self.fwd_passes)

                eval_loss += result['loss']
                eval_acc  += result['accuracy']
                
                # Add the local result to the final result
                indices.extend(eval_indices.numpy())
                labels.extend(self.labels.numpy())
                predictions.extend(result['predicted_labels'].detach().cpu().numpy())
                softmaxes.extend(result["softmax"].detach().cpu().numpy())

                if self.mc_dropout:
                    uncertainties.extend(result["uncertainty"].detach().cpu().numpy())
           
                print("eval_iteration : " + str(it) + " eval_loss : " + str(result["loss"]) + " eval_accuracy : " + str(result["accuracy"]))
            
                eval_iterations += 1
        
        # convert arrays to torch tensors
        print("loss : " + str(eval_loss/eval_iterations) + " accuracy : " + str(eval_acc/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
        accuracy = np.array([eval_acc])

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss, "eval_acc":accuracy}
        
        indices     = np.array(indices)
        labels      = np.array(labels)
        predictions = np.array(predictions)
        softmaxes   = np.array(softmaxes)
        uncertainties = np.array(uncertainties)
        
        local_eval_results_dict = {"indices":indices, "labels":labels, "predictions":predictions, "softmaxes":softmaxes, "uncertainties":uncertainties}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                predictions = np.array(global_eval_results_dict["predictions"].cpu())
                softmaxes   = np.array(global_eval_results_dict["softmaxes"].cpu())
                uncertainties = np.array(global_eval_results_dict["uncertainties"].cpu())
        
        if self.rank == 0:
#            print("Sorting Outputs...")
#            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", indices)#sorted_indices)
            np.save(self.dirpath + "labels.npy", labels)#[sorted_indices])
            np.save(self.dirpath + "predictions.npy", predictions)#[sorted_indices])
            np.save(self.dirpath + "softmax.npy", softmaxes)#[sorted_indices])
            if self.mc_dropout:
                np.save(self.dirpath + "uncertainties.npy", uncertainties)

            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
            val_acc = np.sum(local_eval_metrics_dict["eval_acc"])

            print("\nAvg eval loss : " + str(val_loss/val_iterations),
                  "\nAvg eval acc : "  + str(val_acc/val_iterations))


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            # TODO: check that BN layers are in eval mode
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
    # ========================================================================
    # Saving and loading models

    def save_state(self, best=False):
        """
        Save model weights to a file.
        
        Args:
            best    ... if true, save as best model found, else save as checkpoint
        
        Outputs:
            dict containing iteration, optimizer state dict, and model state dict
            
        Returns: filename
        """
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.model_accs.state_dict()
        
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_best_state(self, placeholder):
        """
        Restore model using best model found in current directory

        Args:
            placeholder     ... extraneous; hydra configs are not allowed to be empty

        Outputs: model params are now those loaded from best model file
        """
        best_validation_path = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     "BEST",
                                     ".pth")

        self.restore_state_from_file(best_validation_path)
    
    def restore_state(self, restore_config):
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """
        Restore model using weights stored from a previous run
        
        Args: 
            weight_file     ... path to weights to load
        
        Outputs: model params are now those loaded from file
        """
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])
            
            # if optim is provided, load the state of the optim
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # load iteration count
            self.iteration = checkpoint['global_step']
        
        print('Restoration complete.')
