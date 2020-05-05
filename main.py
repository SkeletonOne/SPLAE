import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

from data_processing import TumorDataset, get_indices
from models.u_net import U_Net
from models.r2u_net import R2U_Net
from models.attu_net import AttU_Net
from models.r2attu_net import R2AttU_Net
from models.nestedu_net import NestedUNet
from utils.generate_2D_imgs import generate_2D_imgs
from utils.dice_coefficient import dice_coefficient
from utils.visualize_result import visualize_result
from loss.dice_loss import BinaryDiceLoss, CosDiceLoss, DiceLoss
from tools.visualize_gt import visualize_gt

#################################### Hyper params start ##################################################
################ DEVICE ##########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################ PATHS ###########
# path for .mhd
file_path = '/content/dataset'
# path for .png
save_path = './imgs/'
model_save_path = './saved_models/'
################ DATA ############
train_num, val_num, test_num = 2500, 127, 127
do_smooth = False
do_hist_equalize = False
do_normalize = False
do_data_augmentation = False
################ MODEL ###########
models = [U_Net(), R2U_Net(), AttU_Net(), R2AttU_Net(), NestedUNet()]
model_used = models[0]
print_model = True
BATCH_SIZE = 8
epochs = 60
learning_rate = 1e-4
loss = 'bce'
#################################### Hyper params end ####################################################

print('\033[1;36mComputation Details: \033[0m')
if device == torch.device('cpu'):
    print(f'\tNo GPU available. Using CPU.')
else:
    print(f'\tDevice Used: ({device})  {torch.cuda.get_device_name(torch.cuda.current_device())}\n')

print('\033[1;36mPackages Used Versions: \033[0m')
print(f'\tPytorch Version: {torch.__version__}')

if not os.path.isdir(save_path):
    os.makedirs(save_path)
    print('\033[1;35mGenerating .png imgs from .mhd files. \033[0m')
    generate_2D_imgs(file_path, save_path, train_num, do_smooth, do_hist_equalize, do_normalize, do_data_augmentation)
    print('\033[1;35mGenerate finished. \033[0m')
else:
    print('Training images already exist. No need to generate them again.')

# Dataset folder used
DATASET_PATH = os.path.join(save_path)
tumor_dataset = TumorDataset(DATASET_PATH)
train_indices, validation_indices, test_indices = get_indices(train_num, val_num, test_num)
train_sampler, validation_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(validation_indices), SubsetRandomSampler(test_indices)
trainloader = torch.utils.data.DataLoader(tumor_dataset, BATCH_SIZE, sampler = train_sampler)
validationloader = torch.utils.data.DataLoader(tumor_dataset, 1, sampler = validation_sampler)
testloader = torch.utils.data.DataLoader(tumor_dataset, 1, sampler = test_sampler)

print('Number of files in the train set: %s \nNumber of files in the validation set: %s \nNumber of files in the test set: %s' \
      % (len(train_indices), len(validation_indices), len(test_indices)))

visualize_gt(DATASET_PATH)
print('Visualized data has been saved to Visualized_data.png')

unet_model = None
unet_classifier = None
if loss == 'bce':
    criterion = nn.BCELoss()
elif loss == 'dice':
    criterion = BinaryDiceLoss()
elif loss == 'cosdice':
    criterion = CosDiceLoss()
else:
    raise NotImplementedError("The loss is not implemented: %s"(loss))

#### If you want to see the training trend within each epoch, you can change mini_batch to a positive integer 
#### that is no larger than the number of batches per epoch.
mini_batch = 100

# Define where to save the model parameters.
os.makedirs(model_save_path, exist_ok = True)

# New model is created.
unet_model = model_used.to(device)

if print_model:
    print(unet_model)

# Training session history data.
history = {'train_loss': list(), 'validation_loss': list()}

# For save best feature. Initial loss taken a very high value.
last_score = 0

# Optimizer used for training process. Adam Optimizer.
optimizer = optim.Adam(unet_model.parameters(), lr = learning_rate)

# Reducing LR on plateau feature to improve training.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.85, patience = 10, verbose = True)

print('\033[1;34mStarting Training Process \033[0m')

assert validationloader.batch_size == 1

# Epoch Loop
for epoch in range(epochs):
    
    #################################### Train ####################################################
    unet_model.train()
    start_time = time()
    # Training a single epoch
    train_epoch_loss, train_batch_loss, batch_iteration = 0, 0, 0
    validation_score, validation_loss = 0, 0

    for batch, data in enumerate(trainloader):
        # Keeping track how many iteration is happening.
        batch_iteration += 1
        # Loading data to device used.
        image = data['image'].to(device)
        mask = data['mask'].to(device)
        # for img in mask:
        #   for c in img:
        #     for r in c:
        #       for co in r:
        #         if co != 0:
        #           print(co)
        # Clearing gradients of optimizer.
        optimizer.zero_grad()
        # Calculation predicted output using forward pass.
        output = unet_model(image)
        # Calculating the loss value.
        loss_value = criterion(output, mask)
        # Computing the gradients.
        loss_value.backward()
        # Optimizing the network parameters.
        optimizer.step()
        # Updating the running training loss
        train_epoch_loss += loss_value.item()
        train_batch_loss += loss_value.item()

        # Printing batch logs if any. Useful if you want to see the training trends within each epoch.
        if mini_batch:
            if (batch + 1) % mini_batch == 0:
                train_batch_loss = train_batch_loss / (mini_batch * trainloader.batch_size)
                print(
                    f'    Batch: {batch + 1:2d},\tBatch Loss: {train_batch_loss:.7f}')
                train_batch_loss = 0

    train_epoch_loss = train_epoch_loss / (batch_iteration * trainloader.batch_size)
    
    ################################### Validation ##################################################
    unet_model.eval()
    # To get data in loops.
    batch_iteration = 0

    for batch, data in enumerate(validationloader):
        # Keeping track how many iteration is happening.
        batch_iteration += 1
        # Data prepared to be given as input to model.
        image = data['image'].to(device)
        
        mask = data['mask'].to(device)

        # Predicted output from the input sample.
        mask_prediction = unet_model(image)
        
        # comput validation loss
        loss_value = criterion(mask_prediction, mask)
        validation_loss += loss_value.item()
        
        # Threshold elimination.
        mask_prediction = (mask_prediction > 0.5)
        mask_prediction = mask_prediction.cpu().numpy()
        mask = mask.cpu().numpy()

        mask = np.resize(mask, (1, 512, 512))
        mask_prediction = np.resize(mask_prediction, (1, 512, 512))
        # Calculate the dice score for original and predicted image mask.
        validation_score += dice_coefficient(mask_prediction, mask)

    # Calculating the mean score for the whole validation dataset.
    unet_val = validation_score / batch_iteration
    validation_loss = validation_loss / batch_iteration
    
    # Collecting all epoch loss values for future visualization.
    history['train_loss'].append(train_epoch_loss)
    history['validation_loss'].append(validation_loss)
    
    # Reduce LR On Plateau
    scheduler.step(validation_loss)

    time_taken = time() - start_time
    
    # Training Logs printed.
    print(f'Epoch: {epoch + 1:3d},  ', end = '')
    print(f'train Loss: {train_epoch_loss:.5f},  ', end = '')
    print(f'validation Loss: {validation_loss:.5f},  ', end = '')
    print(f'validation score: {unet_val:.5f},  ', end = '')

    for pg in optimizer.param_groups:
        print('current lr: ', pg['lr'], ', ', end = '')
    print(f'Time: {time_taken:.2f} s', end = '')

    # Save the model every epoch.
    current_epoch_model_save_path = os.path.join(model_save_path, 'Basic_Unet_epoch_%s.pth' % (str(epoch).zfill(3)))
    torch.save(unet_model.state_dict(), current_epoch_model_save_path)
    
    # Save the best model (determined by validation score) and give it a unique name.
    best_model_path = os.path.join(model_save_path, 'Basic_Unet_best_model.pth')
    if  last_score < unet_val:
        torch.save(unet_model.state_dict(), best_model_path)
        last_score = unet_val
        print(f'\tBest model saved at score: {unet_val:.5f}')
    else:
        print()

print(f'Training Finished after {epochs} epoches')

################################### Figure out training history ##################################################

plt.figure(figsize=(20, 10))
plt.title('Loss Over Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
train_curve = plt.plot(history['train_loss'], marker = 'o', label = 'Train loss')
validation_curve = plt.plot(history['validation_loss'], marker = 'o', label = 'Validation loss')
plt.legend(fontsize = 15)
plt.savefig('loss.png')
print('Training history has been saved to loss.png')

################################### Test  ##################################################

print('\033[1;34mStarting Test Process \033[0m')
# Load the unet model at its prime (when it performed the best on the validation set).
state_dict = torch.load(os.path.join(model_save_path, 'Basic_Unet_best_model.pth'))
unet_model.load_state_dict(state_dict)

# Testing process on test data.
unet_model.eval()
# Getting test data indices for dataloading
test_data_indexes = test_indices
# Total testing data used.
data_length = len(test_data_indexes)
# Score after testing on dataset.
mean_test_score = 0

for batch, data in enumerate(testloader):
    # Data prepared to be given as input to model.
    image = data['image'].to(device)
    mask = data['mask']

    # Predicted output from the input sample.
    mask_prediction = unet_model(image).cpu()
    # Threshold elimination.
    mask_prediction = (mask_prediction > 0.5)
    mask_prediction = mask_prediction.numpy()

    mask = np.resize(mask, (1, 512, 512))
    mask_prediction = np.resize(mask_prediction, (1, 512, 512))

    # Calculating the dice score for original and predicted mask.
    mean_test_score += dice_coefficient(mask_prediction, mask)

# Calculating the mean score for the whole test dataset.
unet_score = mean_test_score / data_length
# Putting the model back to training mode.
print(f'\nDice Score {unet_score}\n')

################################### Visualize predictions  ##################################################

for example_index in range(10):
    # The purpose of image_index is to make sure we truly pick from the test set.
    image_index = test_indices[example_index]
    sample = tumor_dataset[image_index]
    threshold = 0.5

    unet_model.eval()
    image = sample['image'].numpy()
    mask = sample['mask'].numpy()

    image_tensor = torch.Tensor(image)
    image_tensor = image_tensor.view((-1, 1, 512, 512)).to(device)
    output = unet_model(image_tensor).detach().cpu()
    output = (output > threshold)
    output = output.numpy()

    # image(numpy.ndarray): 512x512 Original brain scanned image.
    image = np.resize(image, (512, 512))
    # mask(numpy.ndarray): 512x512 Original mask of scanned image.
    mask = np.resize(mask, (512, 512))
    # output(numpy.ndarray): 512x512 Generated mask of scanned image.
    output = np.resize(output, (512, 512))
    # score(float): Sørensen–Dice Coefficient for mask and output. Calculates how similar are the two images.
    d_score = dice_coefficient(output, mask)

    title = f'Name: {image_index}.png   Dice Score: {d_score:.5f}'
    # save_path = os.path.join('images',f'{d_score:.5f}_{image_index}.png')
    visualize_result(image, mask, output, title, example_index)
    print('Predictions has been saved to predictions+(num_of_pred).png')