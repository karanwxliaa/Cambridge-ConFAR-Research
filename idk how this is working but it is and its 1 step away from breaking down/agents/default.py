from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, accuracy_au,accuracy_av, AverageMeter, Timer
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torchmetrics.regression import ConcordanceCorrCoef
from sklearn.metrics import mean_squared_error

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): task=str,lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.


        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        self.model = self.create_model() #check if we need to create or use existing model 
    

      
        if agent_config['gpuid'][0] > 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        
        self.reset_optimizer = False
        self.valid_out_dim = 0  # Default: 0

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.1)

    def create_model(self):
        cfg = self.config
        cfg['model_type'] = cfg['model_type']
        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features
        #print("nfeat = ",n_feat)

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            
            model.last[task] = nn.Linear(n_feat,out_dim) #check for key and assign the output value
            #print("model.last for task ",task," is ",n_feat," & ",out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)



    def predict(self, inputs):
        #cfg = self.config
        self.model.eval()
        
        out = self.forward(inputs.float())
        
        for t in out.keys():
            out[t] = out[t].detach()
        return out


    def validation_av(self, dataloader):
        batch_timer = Timer()
        ccc_meter = AverageMeter()
        f1_meter = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        output_list = []
        target_list = []

        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            output = self.predict(input)
            predicted = output[task[0]]  # Assuming 'AV' is the key for this task
            output_list.append(predicted)
            target_list.append(target)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            ccc = accumulate_ccc(output, target, task, ccc_meter)
            
        outputs = torch.cat(output_list, dim=0)
        targets = torch.cat(target_list, dim=0)

        mse = mean_squared_error(
            torch.Tensor.cpu(targets).detach().numpy(),
            torch.Tensor.cpu(outputs).detach().numpy()
        )
        print("MSE: " + str(mse))

        self.train(orig_mode)

        self.log(' * Val CCC {ccc.avg:.3f}, MSE Score {mse:.3f}, Total time {time:.2f}'
                .format(ccc=ccc_meter, mse=mse, time=batch_timer.toc()))

        return ccc.avg, mse

        

    def validation_fer(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        output_list = []
        target_list = []
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            
            output = self.predict(input)
            predicted = F.softmax(output[task[0]], dim=1)  #changed
            _, predicted = torch.max(predicted, 1)            
            output_list.append(predicted)
            target_list.append(target)   
  
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)
        outputs = torch.cat(output_list, dim=0)
        targets = torch.cat(target_list, dim=0) 
        f1 = f1_score(torch.Tensor.cpu(targets).detach().numpy(), torch.Tensor.cpu(outputs).detach().numpy(), average='weighted')
        print("f1 score: " + str(f1))
        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg, f1
    
    
    def validation_au(self, dataloader):
        import numpy as np
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        output_list = []
        target_list = []

        for i, (input, target, task) in enumerate(dataloader):
  

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            output = self.predict(input)

            outputs = output["AU"]
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0  
            output["AU"] = outputs

            output_list.append(outputs)
            target_list.append(target)   
  
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc_au(output, target, task, acc)

        outputs = torch.cat(output_list, dim=0)
        targets = torch.cat(target_list, dim=0) 
        N_val,C_val = targets.shape
        f1_val = np.zeros((C_val,1))
        for kk in range(C_val):    
            f1_val[kk] = f1_score(targets[kk].cpu().detach().numpy(), outputs[kk].cpu().detach().numpy(), average="binary")
        print("f1 score: " + str(f1_val.mean()))
        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg, f1_val.mean()
    
    def validation(self, dataloader,val_task):
        if val_task=="FER":
            return self.validation_fer(dataloader)
        if val_task=="AV":
            return self.validation_av(dataloader)
        if val_task=="AU":
            return self.validation_au(dataloader)

    def criterion(self, preds, targets, tasks, **kwargs):

        loss = 0
        for t,t_preds in preds.items():
            inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
            if len(inds)>0:
                t_preds = t_preds[inds]
                t_target = targets[inds]
                #print(t_preds,t_target,sep='\n')
                #print("Self crit fun = ", self.criterion_fn)
                #print("self.criterion_fn",  self.criterion_fn)
                loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
        loss /= len(targets)  # Average the total loss by the mini-batch size
        
        return loss

    def update_model(self, inputs, targets, tasks):

        out = self.forward(inputs)
        #print("out = ", out)
        loss = self.criterion(out, targets, tasks)
        #print("loss in update model = ",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def learn_batch(self, train_loader, val_loader, task_no):
        cfg = self.config

        if cfg["tasks"][task_no]=="FER":
            self.criterion_fn = nn.CrossEntropyLoss()
        elif cfg["tasks"][task_no]=="AU":
            self.criterion_fn = nn.MultiLabelMarginLoss()
        elif cfg["tasks"][task_no]=="AV":
            # self.criterion_fn = ConcordanceCorrCoef(num_outputs=2)
            self.criterion_fn = torch.nn.MSELoss()



        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
            
        flag = 0
        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()
            
            if (flag==1):
                break
            
            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task_name) in enumerate(train_loader):
                input = input.float()
                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()
              
#                if(epoch == 2 and task[0] == "2"):
#                    flag = 1
                #print("passing",input, target, task_name,sep=" | ")
                loss, output = self.update_model(input, target, task_name)
                input = input.detach()
                target = target.detach()

             
                if cfg["tasks"][task_no] == 'FER':
                    acc = accumulate_acc(output, target, task_name, acc)

                if cfg["tasks"][task_no] == 'AV':
                    acc = accumulate_ccc(output, target, task_name, acc)

                elif cfg["tasks"][task_no] == 'AU':
                    # measure accuracy and record loss
                    outputs = output["AU"]
                    outputs[outputs >= 0.5] = 1
                    outputs[outputs < 0.5] = 0   
                    output["AU"] = outputs  
                    acc = accumulate_acc_au(output, target, task_name, acc)
                


                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # UPADTE THIS FOR FER AND AU
            if val_loader != None:
                if cfg["tasks"][task_no] == 'FER':
                    axx, _ = self.validation_fer(val_loader)
                elif cfg["tasks"][task_no] == 'AV':
                    axx, _ = self.validation_av(val_loader)
                elif cfg["tasks"][task_no] == 'AU':
                    axx, _ = self.validation_au(val_loader)
                
            if axx > 90:
                break
    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def accumulate_acc(output, target, task, meter):

    for t, t_out in output.items():
        inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
        if len(inds) > 0:
            t_out = t_out[inds]
            t_target = target[inds]
            meter.update(accuracy(t_out, t_target), len(inds))

    return meter




def accumulate_ccc(output, target, task, meter):
    
    for t, t_out in output.items():
        inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched a specific task
        if len(inds) > 0:
            t_out = t_out[inds]
            t_target = target[inds]
            meter.update(accuracy_av(t_out, t_target), len(inds))

    return meter


def accumulate_acc_au(output, target, task, meter):

    for t, t_out in output.items():
        inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
        if len(inds) > 0:
            t_out = t_out[inds]
            t_target = target[inds]
            meter.update(accuracy_au(t_out, t_target), len(inds))
    return meter
