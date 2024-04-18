import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L (int): input feature dimension
        D (int): hidden layer dimension
        dropout (float): whether to use dropout (p = 0.25)
        n_classes (int): number of classes 
    """

    def __init__(self, L = 1024, D = 256, dropout = 0., n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout), 
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    
    args:
        L (int): input feature dimension
        D (int): hidden layer dimension
        dropout (float): whether to use dropout (p = 0.25)
        n_classes (int): number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = 0., n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

class CLAM_SB_ADV(nn.Module):
    """
    args:
        gate (bool): whether to use gated attention network
        size_arg (str): config for network size
        dropout (float): whether to use dropout
        k_sample (int): number of positive/neg patches to sample for instance-level training
        dropout (float): whether to use dropout (p = 0.25)
        n_classes (int): number of classes 
        instance_loss_fn (PyTorch loss function): loss function to supervise instance-level training
        subtyping (bool): whether it's a subtyping problem
        n_classes_adv (int): number of classes for adversarial classifier
    """
    def __init__(self, gate = True, in_dim = 1024, size_arg = "small", dropout = 0.25, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, n_classes_adv=3):
        super(CLAM_SB_ADV, self).__init__()
        self.size_dict = {"small": [in_dim, 512, 256], "big": [in_dim, 512, 384]}
        size = self.size_dict[size_arg]
        
        self.fc = nn.Sequential(*[nn.Linear(size[0], size[1]), 
                   nn.ReLU(),
                   nn.Dropout(dropout)])

        if gate:
            self.attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        self.classifiers = nn.Linear(size[1], n_classes)
        self.classifiers_adv = nn.Linear(size[1], n_classes_adv)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = self.fc(h)
        A = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        logits_adv = self.classifiers_adv(M)

        results_dict = {'logits': logits, 'logits_adv': logits_adv, 'A': A_raw}
        
        if instance_eval:
            results_dict.update({'instance_loss': total_inst_loss})
        
        if return_features:
            results_dict.update({'features': M})

        return results_dict