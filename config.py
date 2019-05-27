import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
# print('device: ' + str(device))

im_size = 224
# Data parameters
num_classes = 9935

pickle_file = ''