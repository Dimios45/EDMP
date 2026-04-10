import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from diffusion.models.blocks import *

class TemporalUNet(nn.Module):

    def __init__(self, model_name, input_dim, time_dim, device, dims = (32, 64, 128, 256)):

        super(TemporalUNet, self).__init__()

        dims = [input_dim, *dims]  # length of dims is 5

        # Initial Time Embedding:
        self.time_embedding = TimeEmbedding(time_dim, device)

        # Down Sampling:
        self.down_samplers = nn.ModuleList([])
        for i in range(len(dims) - 2):      # Loops 0, 1, 2
            self.down_samplers.append(DownSampler(dims[i], dims[i+1], time_dim))
        self.down_samplers.append(DownSampler(dims[-2], dims[-1], time_dim, is_last = True))  # 3 -> 4

        # Middle Block:
        self.middle_block = MiddleBlock(dims[-1], time_dim)

        # Up Sampling:
        self.up_samplers = nn.ModuleList([])
        for i in range(len(dims) - 1, 1, -1):  # Loops 4, 3, 2  since the last one is a seperate convolution
            self.up_samplers.append(UpSampler(dims[i-1], dims[i], time_dim))

        # Final Convolution:
        self.final_conv = nn.Sequential(Conv1dBlock(dims[1], dims[1], kernel_size = 5),
                                        nn.Conv1d(dims[1], input_dim, kernel_size = 1))
        
        self.model_name = model_name
        self.weights_path = os.path.join(self.model_name, "weights_latest.pt")
        self.losses_path = os.path.join(self.model_name, "losses.npy")

        if not os.path.exists(model_name):
            os.mkdir(model_name)
            self.losses = np.array([])
        elif os.path.exists(self.weights_path) and os.path.exists(self.losses_path):
            self.load()
        else:
            self.losses = np.array([])

        _ = self.to(device)

    def forward(self, x, t):
        """
        x => Tensor of size (batch_size, channels, horizon)
        t => Integer representing the diffusion timestep of x
        """

        input_horizon = x.shape[2]

        # Get the time embedding from t:
        time_emb = self.time_embedding(t)

        # Down Sampling Layers:
        h_list = []
        for i in range(len(self.down_samplers)):
            x, h = self.down_samplers[i](x, time_emb)
            h_list.append(h)

        # Middle Layer:
        x = self.middle_block(x, time_emb)

        # Up Sampling Layers:
        for i in range(len(self.up_samplers)):
            h_temp = h_list.pop()
            x = self.up_samplers[i](x, h_temp, time_emb)
            target_horizon = h_list[-1].shape[2] if h_list else input_horizon
            x = self._match_horizon(x, target_horizon)

        # Final Convolution
        out = self.final_conv(x)

        return out

    @staticmethod
    def _match_horizon(x, target_horizon):

        horizon = x.shape[2]
        if horizon == target_horizon:
            return x
        if horizon > target_horizon:
            return x[:, :, :target_horizon]
        return F.pad(x, (0, target_horizon - horizon))

    def save(self):

        torch.save(self.state_dict(), self.weights_path)
        np.save(self.losses_path, self.losses)

    def save_checkpoint(self, checkpoint):
        
        torch.save(self.state_dict(), self.model_name + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_name + "/latest_checkpoint.npy", checkpoint)
    
    def load(self):

        self.losses = np.load(self.losses_path)
        self.load_state_dict(torch.load(self.weights_path))
        print("Loaded Model at " + str(self.losses.size) + " epochs")

    def load_checkpoint(self, checkpoint):

        _ = input("Press Enter if you are running the model for inference, or Ctrl+C\n(Never load a checkpoint for training! This will overwrite progress)")
        
        latest_checkpoint = np.load(self.model_name + "/latest_checkpoint.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_" + str(checkpoint) + ".pt"))
        self.losses = np.load(self.model_name + "/losses.npy")[:checkpoint]

    

            
