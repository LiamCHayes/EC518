import torch
import torch.nn as nn
import numpy as np

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.gpu = torch.device('cuda')
        self.model = torch.nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(140448, 2048),
                nn.Dropout(p=0.5),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(2048, 9),
                nn.Softmax(dim=1)
                ).to(self.gpu)


    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        observation = self.model(observation)
        return observation

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        action_class = torch.from_numpy(np.zeros(shape=(actions.size(dim=0), 9)))
        for r in range(actions.size(dim=0)): 
            # Test if accelerating, none, or break
            if actions[r][1].item() > 0:
                i = 0
            elif actions[r][2].item() > 0:
                i = 3
            else:
                i = 6
        
            # Test if straight, left, or right
            if actions[r][0].item() < -0.02: # Left
                i = i
            if actions[r][0].item() > 0.02:
                i = i+1
            else:
                i = i+2
            
            action_class[r][i] = 1    

        action_class = action_class.type(torch.FloatTensor)
        return action_class.to(self.gpu)

    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steering, accelerating, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        pass


