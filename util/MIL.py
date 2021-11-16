import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, model):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor = model
        self._to_linear = model._to_linear
        
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, self.L),
            nn.ReLU())
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)
        H = H.view(-1, self._to_linear)
        H = self.fc(H)  # [b x L]

        A = self.attention(H)  # [b x K]
        A = torch.transpose(A, 1, 0)  # [K x b]
        A = F.softmax(A, dim=1)  # softmax over b
            
        M = torch.mm(A, H)  # [K x L]

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
        
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return loss
    
    
class GatedAttention(nn.Module):
    def __init__(self, model):
        super(GatedAttention, self).__init__()
        self.name = 'Gated Attention'
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor = model
        self._to_linear = model._to_linear
        
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, self.L),
            nn.ReLU())

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh())

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)
        H = H.view(-1, self._to_linear)
        H = self.fc(H)  # [b x L]

        A_V = self.attention_V(H)  # [b x D]
        A_U = self.attention_U(H)  # [b x D]
        A = self.attention_weights(A_V * A_U) # element wise multiplication -> [b x K]
        A = torch.transpose(A, 1, 0)  # [K x b]
        A = F.softmax(A, dim=1)  # softmax over b

        M = torch.mm(A, H)  # [K x L]

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return loss
    
    
class MIL_pool(nn.Module):
    def __init__(self, model, operator='mean'):
        super(MIL_pool, self).__init__()
        self.L = 500 
        if operator == 'mean':
            self.operator = 'mean'
        elif operator == 'max':
            self.operator = 'max'    
        else:
            raise NotImplementedError('Operator not supported: {}'.format(operator))

        self.name = 'MIL pool ' + self.operator
        self.feature_extractor = model
        self._to_linear = model._to_linear

        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, self.L),
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)
        
        # prepNN
        H = self.feature_extractor(x)
        H = H.view(-1, self._to_linear)
        H = self.fc(H)  # [b x L]
        
        # aggregate function
        if self.operator == 'mean':
            M = torch.mean(H, 0)
        else:
            M = torch.amax(H, 0)
          
        # afterNN
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
        
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return loss