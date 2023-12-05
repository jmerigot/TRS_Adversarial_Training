#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
import torchvision.transforms as transforms
from tqdm import tqdm
import random


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 512

'''Basic neural network architecture (from pytorch doc).'''
class Neural_Network(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))




class Net(nn.Module):
    
    model_file = "models/trs_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    models = []

    pgd_linf_model_path = "models/adv_model.pth"
    pgd_l2_model_path = "models/pgd_l2_model.pth"
    fgsm_model_path = "models/FGSM_model.pth"
    model_paths = [pgd_linf_model_path, pgd_l2_model_path, fgsm_model_path]

    for model_path in model_paths:
        model = Neural_Network().to(device)
        model.load(model_path)
        models.append(model)

    def __init__(self, models=models):
        super(Net, self).__init__()
        self.models = models
    
    def forward(self, x):
        outputs = 0
        i = 1
        for model in self.models:
            outputs += torch.exp(model(x))
            #print(model(x)[0])
            max_ind= torch.argmax(model(x)[0])
            #print(f"L'output du model {i} est de {max_ind} ")
            i += 1
        output = outputs / len(self.models)
        output = torch.clamp(output, min=1e-40)
        return torch.log(output)
        


    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))



def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))
    
def update_eps_alpha(epoch, num_epochs, eps, final_eps, alpha, final_alpha):
    scale = epoch / (2 * num_epochs)
    new_epsilon = (final_eps - eps) * scale + eps
    new_alpha = (final_alpha - alpha) * scale + alpha
        
    return new_epsilon, new_alpha
    
def train_model_adversarial(net, train_loader, pth_filename, num_epochs, 
                            eps=0.03, alpha=0.01, iters=20, step_size=1, gamma=1, adv_prob = 0.2):
    print("Starting training with adversarial examples")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    final_eps = 0.08
    final_alpha = 0.03

    for epoch in tqdm(range(num_epochs)):  
        
        eps, alpha = update_eps_alpha(epoch, num_epochs, eps, final_eps, alpha, final_alpha)
        
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0)):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Decide whether to use adversarial examples or not
            if random.random() < adv_prob:
                # Generate adversarial examples
                input_set = pgd_attack(net, inputs, labels, eps, alpha, iters)
            else:
                input_set = inputs

            # Train on the chosen set (adversarial or natural)
            optimizer.zero_grad()
            outputs = net(input_set)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            """
            # Generate adversarial examples
            #adv_inputs = pgd_attack(net, inputs, labels, eps, alpha, iters)
            
            # Train on both natural and adversarial examples
            for input_set in [inputs, adv_inputs]:
                optimizer.zero_grad()
                outputs = net(input_set)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            """

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
        scheduler.step()

    net.save(pth_filename)
    print('Finished Adversarial Training')

    
def pgd_attack(model, images, labels, eps, alpha, iters):
    original_images = images.clone().detach()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
        
    return images

def pgd_attack_l2(model, images, labels, eps, alpha, iters):
    original_images = images.clone().detach().to(device)  # Only clone images once
    images = images.clone().detach().to(device)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.nll_loss(outputs, labels.to(device))  # Move labels to device without cloning
        model.zero_grad()
        loss.backward()
        grad = images.grad.data

        # L2 norm
        norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-8  # Add small constant to avoid division by zero
        normed_grad = grad / norm

        adv_images = images + alpha * normed_grad
        delta = adv_images - original_images
        delta = torch.clamp(delta, min=-eps, max=eps)

        # Project back into L2 ball
        mask = delta.view(delta.shape[0], -1).norm(p=2, dim=1) <= eps
        scaling_factor = delta.view(delta.shape[0], -1).norm(p=2, dim=1)
        scaling_factor[mask] = eps
        delta = delta * (eps / scaling_factor.view(-1, 1, 1, 1))

        images = original_images + delta
        images = torch.clamp(images, min=0, max=1).detach_()  # In-place clamp and detach

    return images




def Cosine(g1, g2):
	return torch.abs(F.cosine_similarity(g1, g2)).mean() 

def Magnitude(g1):
	return (torch.sum(g1**2,1)).mean() * 2

def TRS_training(loader, valid_loader, models, num_epochs, save_path):
    criterion = nn.CrossEntropyLoss()

    param = list(models[0].parameters())
    for i in range(1, len(models)):
        param.extend(list(models[i].parameters()))

    lr = 0.001
    gamma=0.5
    step_size=11

    optimizer = optim.Adam(param, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_eps = 0.03
    start_alpha = 0.01
    final_eps = 0.08
    final_alpha = 0.03

    for i in range(len(models)):
        models[i].train()
        models[i].requires_grad = True
    
    accuracy_test = [0]*3
    for i, (inputs, targets) in enumerate(valid_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        for j in range(len(models)):
            adv_images = pgd_attack(models[j], inputs, targets, start_eps, start_alpha, iters=20).detach()
            adv_outputs = models[j](adv_images).to(device)
            _, predicted = torch.max(adv_outputs.data, 1)
            accuracy_test[j] += (predicted == targets).sum().item()/targets.size(0)
    for j in range(len(models)):
        print(f"avant epoch : accuracy model {j} : {accuracy_test[j]/len(valid_loader)}")

    for epoch in tqdm(range(num_epochs)):  
        eps, alpha = update_eps_alpha(epoch, num_epochs, start_eps, final_eps, start_alpha, final_alpha)
        epoch_loss = 0
        valid_epoch_loss = 0
        accuracy_test = [0]*3
       

        for i, (inputs, targets) in tqdm(enumerate(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            loss_std = 0
            for j in range(len(models)):
                model_output = models[j](inputs).to(device)
                loss = criterion(model_output, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

            cos_loss, smooth_loss = 0, 0

            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])

            cos_loss = (cos01 + cos02 + cos12) / 3.

            N = inputs.shape[0] // 2
            clean_inputs = inputs[:N].detach()
            adv_linf_inputs = pgd_attack(models[0], inputs[N:], targets[N:], eps, alpha, iters=20).detach()
            #adv_linf_inputs = pgd_attack(models[0], inputs[N: 2*N], targets[N: 2*N], eps, alpha, iters=10).detach()
            #adv_l2_inputs = pgd_attack_l2(models[1], inputs[2*N :], targets[2*N :], eps, alpha, iters=10).detach()

            adv_x = torch.cat([clean_inputs, adv_linf_inputs])#, adv_l2_inputs])
            adv_x.requires_grad = True

            for j in range(len(models)):
                output = models[j](adv_x).to(device)
                loss = criterion(output, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

            smooth_loss /= 3

            scale = 1/3
            coeff = 100
            lamda = 2.5
            loss = loss_std + scale * (coeff * cos_loss + lamda * smooth_loss)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          

            ensemble = Net(models)

        scheduler.step()

        for i, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            loss_std = 0
            for j in range(len(models)):
                model_output = models[j](inputs)
                loss = criterion(model_output, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

                
                adv_images = pgd_attack(models[j], inputs, targets, eps, alpha, iters=20).detach()
                adv_outputs = models[j](adv_images).to(device)
                _, predicted = torch.max(adv_outputs.data, 1)
                accuracy_test[j] += (predicted == targets).sum().item()/targets.size(0)



            cos_loss, smooth_loss = 0, 0

            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])

            cos_loss = (cos01 + cos02 + cos12) / 3.

            N = inputs.shape[0] // 2
            clean_inputs = inputs[:N].detach()
            adv_linf_inputs = pgd_attack(models[0], inputs[N:], targets[N:], eps, alpha, iters=20).detach()
            # adv_linf_inputs = pgd_attack(models[0], inputs[N: 2*N], targets[N: 2*N], eps, alpha, iters=20).detach()
            # adv_l2_inputs = pgd_attack_l2(models[1], inputs[2*N :], targets[2*N :], eps, alpha, iters=20).detach()

            adv_x = torch.cat([clean_inputs, adv_linf_inputs])#, adv_l2_inputs])
            adv_x.requires_grad = True

            for j in range(len(models)):
                output = models[j](adv_x)
                loss = criterion(output, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

            smooth_loss /= 3

            scale = 1/3
            coeff = 100
            lamda = 2.5
            valid_loss = loss_std + scale * (coeff * cos_loss + lamda * smooth_loss)
            valid_epoch_loss += valid_loss



        if epoch%1 == 0:
            print(f"Epoch {epoch} : \n Loss = {epoch_loss/len(loader)}")
            print(f"valid_Loss = {valid_epoch_loss/len(valid_loader)}")
            for j in range(len(models)):
                print(f"accuracy model {j} : {accuracy_test[j]/len(valid_loader)}")
  


        
    
    ensemble.save(save_path)
    print('Finished Adversarial Training')


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=16):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=16):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-t', '--trs', action="store_true",
                        help="Using TRS training.")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=20,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)
    
    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar_train = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar_train, valid_size, batch_size=batch_size)
        
        cifar_valid = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
        valid_loader = get_validation_loader(cifar_valid, valid_size)

        #train_model(net, train_loader, args.model_file, args.num_epochs)
        
        if args.force_train:
            train_model_adversarial(net, train_loader, args.model_file, args.num_epochs)
            print("Model save to '{}'.".format(args.model_file))
            
        elif args.trs:
            TRS_training(train_loader, valid_loader, net.models, num_epochs=10, save_path="models/trs_model.pth")
            print("Model save to '{}'.".format(args.model_file))
            

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {} %".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()
