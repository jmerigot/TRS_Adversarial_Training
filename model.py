  #!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
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
    
    model_file = ["models/trained_model_1.pth", "models/trained_model_2.pth", "models/trained_model_3.pth"]
    pretrained_file= ["models/adv_model.pth", "models/pgd_l2_model.pth", "models/FGSM_model.pth"]
    
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''


    def __init__(self):
        super(Net, self).__init__()
        self.models=[]
        for i in range(3):
            self.models.append(Neural_Network().to(device))## On initialise avec 3 models vierges et on les ecrasera si on veut en load d'autres 
    
    def forward(self, x):
        outputs = 0
        i = 1
        for model in self.models:
            outputs += torch.exp(model(x))# L'achitecture imposait d'avoir un log softmax à la fin du reseaux de neuronnes, on veut faire une moyenne de softmax donc on enleve temporairement le log et on le remettra à la fin
        output = outputs / len(self.models)
        output = torch.clamp(output, min=1e-40) #Evite le log(0)
        return torch.log(output) # on remet le log
        


    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        if len(model_file)==len(self.models):
            for i in range(len(self.models)) :
                self.models[i].save(model_file[i]) #On enregistre le model i avec le i-eme nom de model_file
        else : 
            print("Donner en argument du save autant de path que de modèls à  enregistrer")

    def load(self, model_file):
        self.models=[]#On ecrase les derniers les anciens models
        for model_path in model_file:
            model = Neural_Network().to(device)
            model.load(model_path)
            self.models.append(model)

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''
        paths = []

        for path in Net.model_file:
            paths.append(os.path.join(project_dir, path))
        self.load(paths)





def update_eps_alpha(epoch, num_epochs, eps, final_eps, alpha, final_alpha):
    scale = epoch / (2 * num_epochs)
    new_epsilon = (final_eps - eps) * scale + eps
    new_alpha = (final_alpha - alpha) * scale + alpha
        
    return new_epsilon, new_alpha



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



def losses_plot(training_loss, validation_loss, save_path):
    num_epochs = [i+1 for i in range(len(training_loss))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs, training_loss, label='Training Loss')
    plt.plot(num_epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig(save_path, format='png', dpi=300)

    # Close the plot explicitly after saving
    plt.close()


def accuracy_plot_models(valid_accuracy_model_0, valid_accuracy_model_1, valid_accuracy_model_2, valid_accuracy_ensemble_linf, save_path):
    num_epochs = [i+1 for i in range(len(valid_accuracy_model_0))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs, valid_accuracy_model_0, label='Accuracy model 1, attaque PGD linf')
    plt.plot(num_epochs, valid_accuracy_model_1, label='Accuracy model 2, attaque PGD linf')
    plt.plot(num_epochs, valid_accuracy_model_2, label='Accuracy model 3, attaque PGD linf')
    plt.plot(num_epochs, valid_accuracy_ensemble_linf, label='Accuracy ensemble model, attaque PGD Linf')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of models Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig(save_path, format='png', dpi=300)

    # Close the plot explicitly after saving
    plt.close()




def accuracy_plot(valid_accuracy_ensemble_natural, valid_accuracy_ensemble_linf, valid_accuracy_ensemble_l2, save_path):
    num_epochs = [i+1 for i in range(len(valid_accuracy_ensemble_l2))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs, valid_accuracy_ensemble_natural, label='Accuracy ensemble model, non attaqué')
    plt.plot(num_epochs, valid_accuracy_ensemble_l2, label='Accuracy ensemble model, attaque PGD L2')
    plt.plot(num_epochs, valid_accuracy_ensemble_linf, label='Accuracy ensemble model, attaque PGD Linf')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of models Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig(save_path, format='png', dpi=300)

    # Close the plot explicitly after saving
    plt.close()
    


def Cosine(g1, g2):
	return torch.abs(F.cosine_similarity(g1, g2)).mean() 



def Magnitude(g1):
	return (torch.sum(g1**2,1)).mean() * 2





def TRS_training(loader, valid_loader, model, num_epochs, adv_prob, save_path):
    criterion = nn.CrossEntropyLoss()

    param = list(model.models[0].parameters())
    for i in range(1, len(model.models)):
        param.extend(list(model.models[i].parameters()))

    lr = 0.001
    gamma=0.5
    step_size=11

    optimizer = optim.Adam(param, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_eps = 0.03
    start_alpha = 0.01
    final_eps = 0.08
    final_alpha = 0.03
    
    training_loss = []
    validation_loss = []
    valid_accuracy_model_0 = []
    valid_accuracy_model_1 = []
    valid_accuracy_model_2 = []
    valid_accuracy_ensemble_natural = []
    valid_accuracy_ensemble_l2 = []
    valid_accuracy_ensemble_linf = []

    for i in range(len(model.models)):
        model.models[i].train()
        model.models[i].requires_grad = True
    

    for epoch in tqdm(range(num_epochs)):  

        for i in range(len(model.models)):
            model.models[i].train()


        eps, alpha = update_eps_alpha(epoch, num_epochs, start_eps, final_eps, start_alpha, final_alpha)
        epoch_loss = 0
        valid_epoch_loss = 0
        accuracy_test = [0]*3
        accuracy_ensemble_natural = 0
        accuracy_ensemble_linf = 0
        accuracy_ensemble_l2 = 0

       

        for i, (inputs, targets) in tqdm(enumerate(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            if random.random() < adv_prob:
                # Generate adversarial examples
                input_set = pgd_attack_l2(model, inputs, targets, eps, alpha, iters=20)
            else:
                input_set = inputs


            batch_size = inputs.size(0)
            inputs.requires_grad = True
            input_set.requires_grad = True
            grads = []
            loss_std = 0


            for j in range(len(model.models)):
                model_output = model.models[j](input_set).to(device)
                loss = criterion(model_output, targets)
                grad = autograd.grad(loss, input_set, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

            cos_loss, smooth_loss = 0, 0

            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])

            cos_loss = (cos01 + cos02 + cos12) / 3.

            N = inputs.shape[0] // 2
            m = N// 3
    
            clean_inputs = inputs[:N].detach()
            adv_inputs_0 = pgd_attack(model.models[0], inputs[N:N+m], targets[N:N+m], eps, alpha, iters=20).detach()
            adv_inputs_1 = pgd_attack(model.models[1], inputs[N+m:N+2*m], targets[N+m:N+2*m], eps, alpha, iters=20).detach()
            adv_inputs_2 = pgd_attack(model.models[2], inputs[N+2*m:N+3*m], targets[N+2*m:N+3*m], eps, alpha, iters=20).detach()
            #adv_linf_inputs = pgd_attack(model.models[0], inputs[N: 2*N], targets[N: 2*N], eps, alpha, iters=10).detach()
            #adv_l2_inputs = pgd_attack_l2(model.models[1], inputs[2*N :], targets[2*N :], eps, alpha, iters=10).detach()

            adv_x = torch.cat([clean_inputs, adv_inputs_0, adv_inputs_1, adv_inputs_2])# adv_linf_inputs])
            adv_x.requires_grad = True
            targets = targets[:adv_x.shape[0]]

            for j in range(len(model.models)):
                output = model.models[j](adv_x).to(device)
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
          

        for i in range(len(model.models)):
            model.models[i].eval()

        scheduler.step()

        for i, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            loss_std = 0
            for j in range(len(model.models)):
                model_output = model.models[j](inputs)
                loss = criterion(model_output, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

                
                adv_images = pgd_attack(model.models[j], inputs, targets, eps, alpha, iters=20).detach()
                adv_outputs = model.models[j](adv_images).to(device)
                _, predicted = torch.max(adv_outputs.data, 1)
                accuracy_test[j] += (predicted == targets).sum().item()/targets.size(0)


            ensemble_output = model(inputs).to(device)
            _, predicted = torch.max(ensemble_output.data, 1)
            accuracy_ensemble_natural += (predicted == targets).sum().item()/targets.size(0)

            
            adv_images = pgd_attack(model, inputs, targets, eps, alpha, iters=20).detach()
            ensemble_output = model(adv_images).to(device)
            _, predicted = torch.max(ensemble_output.data, 1)
            accuracy_ensemble_linf += (predicted == targets).sum().item()/targets.size(0)

            adv_images = pgd_attack_l2(model, inputs, targets, eps, alpha, iters=20).detach()
            ensemble_output = model(adv_images).to(device)
            _, predicted = torch.max(ensemble_output.data, 1)
            accuracy_ensemble_l2 += (predicted == targets).sum().item()/targets.size(0)



            cos_loss, smooth_loss = 0, 0

            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])

            cos_loss = (cos01 + cos02 + cos12) / 3.

            N = inputs.shape[0] // 2
            m = N// 3

            clean_inputs = inputs[:N].detach()
            adv_inputs_0 = pgd_attack(model.models[0], inputs[N:N+m], targets[N:N+m], eps, alpha, iters=20).detach()
            adv_inputs_1 = pgd_attack(model.models[1], inputs[N+m:N+2*m], targets[N+m:N+2*m], eps, alpha, iters=20).detach()
            adv_inputs_2 = pgd_attack(model.models[2], inputs[N+2*m:N+3*m], targets[N+2*m:N+3*m], eps, alpha, iters=20).detach()
            #adv_linf_inputs = pgd_attack(model.models[0], inputs[N: 2*N], targets[N: 2*N], eps, alpha, iters=10).detach()
            #adv_l2_inputs = pgd_attack_l2(model.models[1], inputs[2*N :], targets[2*N :], eps, alpha, iters=10).detach()

            adv_x = torch.cat([clean_inputs, adv_inputs_0, adv_inputs_1, adv_inputs_2])# adv_linf_inputs])
            adv_x.requires_grad = True
            targets = targets[:adv_x.shape[0]]

            for j in range(len(model.models)):
                output = model.models[j](adv_x)
                loss = criterion(output, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

            smooth_loss /= 3


            valid_loss = loss_std + scale * (coeff * cos_loss + lamda * smooth_loss)
            valid_epoch_loss += valid_loss


        training_loss.append((epoch_loss/len(loader)).cpu().detach().numpy())
        validation_loss.append((valid_epoch_loss/len(loader)).cpu().detach().numpy())
        valid_accuracy_model_0.append(accuracy_test[0]/len(valid_loader))
        valid_accuracy_model_1.append(accuracy_test[1]/len(valid_loader))
        valid_accuracy_model_2.append(accuracy_test[2]/len(valid_loader))
        valid_accuracy_ensemble_natural.append(accuracy_ensemble_natural/len(valid_loader))
        valid_accuracy_ensemble_l2.append(accuracy_ensemble_l2/len(valid_loader))        
        valid_accuracy_ensemble_linf.append(accuracy_ensemble_linf/len(valid_loader))



        if epoch%1 == 0:
            print(f"Epoch {epoch} : \n Loss = {epoch_loss/len(loader)}")
            print(f"valid_Loss = {valid_epoch_loss/len(valid_loader)}")
            for j in range(len(model.models)):
                print(f"accuracy model {j} : {accuracy_test[j]/len(valid_loader)}")
            print(f"accuracy ensemble model natural: {accuracy_ensemble_natural/len(valid_loader)}")
            print(f"accuracy ensemble model l2 : {accuracy_ensemble_l2/len(valid_loader)}")
            print(f"accuracy ensemble model linf : {accuracy_ensemble_linf/len(valid_loader)}")
        

    losses_plot(training_loss, validation_loss, r'plot/losses_plot_vanilla.png')
    accuracy_plot_models(valid_accuracy_model_0, valid_accuracy_model_1, valid_accuracy_model_2,  valid_accuracy_ensemble_linf, r'plot/accuracy_plot_vanilla_models.png')
    accuracy_plot(valid_accuracy_ensemble_natural, valid_accuracy_ensemble_linf, valid_accuracy_ensemble_l2, r'plot/accuracy_plot_vanilla.png')
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




def test_adversarial(net, test_loader, num_samples, eps=0.05, alpha=0.01, iters=20, attack='linf'):
    correct = 0
    total = 0

    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        if attack == 'linf':
            adv_images = pgd_attack(net, images, labels, eps, alpha, iters)
        elif attack == "l2":
            adv_images = pgd_attack_l2(net, images, labels, eps, alpha, iters)
        for _ in range(num_samples):
            outputs = net(adv_images)
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
    parser.add_argument("-p", '--pretrained', action="store_true")# si on veut utiliser des modèles préentrainés pour initialiser le modèle ensemblise 
    parser.add_argument('-t', '--trs', action="store_true",
                        help="Using TRS training.")
    args = parser.parse_args()



    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)
    
    train_transform = transforms.Compose([transforms.ToTensor()]) 
    cifar_train = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
    train_loader = get_train_loader(cifar_train, valid_size, batch_size=batch_size)
    
    cifar_valid = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar_valid, valid_size)
    
    acc = test_natural(net, valid_loader)
    acc_adv_linf = test_adversarial(net, valid_loader, num_samples=1, attack='linf')
    acc_adv_l2 = test_adversarial(net, valid_loader, num_samples=1, attack='l2')

    print("Avant load : Model natural accuracy (valid): {} %".format(acc))
    print("Avant load : Model adversarial accuracy linf (valid): {} %".format(acc_adv_linf))
    print("Avant load :  Model adversarial accuracy l2 (valid): {} %".format(acc_adv_l2))

    #### Model training (if necessary)
    
    
    #Si on veut utiliser des models préentrainés, il faut modifier les paths au début de la définition de la class Net
    if args.pretrained:
        net.load(Net.pretrained_file)

    if args.trs:

        acc = test_natural(net, valid_loader)
        acc_adv_linf = test_adversarial(net, valid_loader, num_samples=1, attack='linf')
        acc_adv_l2 = test_adversarial(net, valid_loader, num_samples=1, attack='l2')

        print("Avant entrainement : Model natural accuracy (valid): {} %".format(acc))
        print("Avant entrainement : Model adversarial accuracy linf (valid): {} %".format(acc_adv_linf))
        print("Avant entrainement : Model adversarial accuracy l2 (valid): {} %".format(acc_adv_l2))

        TRS_training(train_loader, valid_loader, net, num_epochs=6, adv_prob=0.8, save_path=Net.model_file)
        print("Model save to '{}'.".format(Net.model_file))
        net.save(Net.model_file)




    #### Model testing ####
    net = Net()
    net.load(Net.model_file)

    print("Testing with model from '{}'. ".format(Net.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used

    acc = test_natural(net, valid_loader)
    acc_adv_linf = test_adversarial(net, valid_loader, num_samples=1, attack='linf')
    acc_adv_l2 = test_adversarial(net, valid_loader, num_samples=1, attack='l2')

    print("Model natural accuracy (valid): {} %".format(acc))
    print("Model adversarial accuracy linf (valid): {} %".format(acc_adv_linf))
    print("Model adversarial accuracy l2 (valid): {} %".format(acc_adv_l2))

if __name__ == "__main__":
    main()
