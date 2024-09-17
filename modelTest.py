import torch

# Load the checkpoint
checkpoint_path = 'StableDiffusion/stable-diffusion-webui/models/StableDiffusion/model.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract the state dictionary
state_dict = checkpoint['state_dict']

# Print the keys in the state dictionary to understand the layers
for key in state_dict.keys():
    print(key)