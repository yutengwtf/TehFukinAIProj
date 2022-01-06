import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Brain(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(), 
            nn.Linear(in_features=512, out_features=env.action_space.n)
        ).double()
        self.target = copy.deepcopy(self.policy)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, t):
        return self.policy(t)

    def get_action(self, obs):
        ''' No gradient, return action(int) '''
        return torch.argmax(self.policy(obs)).item()

    def get_Q(self, obses):
        ''' With gradient, return Qs(tensor) '''
        return self.policy(obses)

    def get_td(self, obses):
        ''' Without gradient, return Qs_t(tensor)'''
        pass

    def learnable(self):
        return self.policy.parameters()

    def update(self):
        self.target.load_state_dict(self.policy.state_dict())

class TargetNetworkBrain(Brain):
    def __init__(self, env):
        super().__init__(env)

    def get_td(self, obses):
        ''' Without gradient, return Qs_t(tensor)'''
        with torch.no_grad():
            y = self.target(obses)
            return torch.max(y, dim=1)[0]

class DoubleDQNBrain(Brain):
    def __init__(self, env):
        super().__init__(env)

    def get_td(self, obses):
        ''' Without gradient, return Qs_t(tensor)'''
        with torch.no_grad():
            actions_t = torch.argmax(self.policy(obses), dim=1)
            y_Q = self.target(obses)
            Qs_t = torch.gather(y_Q, 1, actions_t.view(-1, 1)).view(-1)
            return Qs_t
            
class DuelingNetworkBrain(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv_ly = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).double()

        self.val_ly = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        ).double()

        self.adv_ly = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=env.action_space.n)
        ).double()

    def forward(self, t):
        charateristic = self.conv_ly(t)
        advantage = self.adv_ly(charateristic)
        value = self.val_ly(charateristic)
        return advantage + value

    def learnable(self):
        return self.parameters()

    def get_action(self, obs):
        ''' No gradient, return action(int) '''
        return torch.argmax(self(obs)).item()

    def get_Q(self, obses):
        ''' With gradient, return Qs(tensor) '''
        return self(obses)

    def get_td(self, obses):
        ''' Without gradient, return Qs_t(tensor)'''
        with torch.no_grad():
            y = self(obses)
            return torch.max(y, dim=1)[0]