import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LowRankRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) / (rank ** 0.5))
        self.V = nn.Parameter(torch.randn(rank, hidden_size) / (hidden_size ** 0.5))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, input, hidden):
        input_part = self.input_proj(input)
        recurrent_part = torch.mm(hidden, self.U)
        recurrent_part = torch.mm(recurrent_part, self.V)
        gate_input = input_part + recurrent_part + self.bias
        new_hidden = torch.tanh(gate_input)
        return new_hidden


class RIMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_mechanisms=4,
                 key_size=16, rank=16, num_heads=1,task_size =0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mechanisms = num_mechanisms
        self.key_size = key_size
        self.rank = rank
        self.num_heads = num_heads

        # Create independent low-rank recurrent networks
        self.mechanisms = nn.ModuleList([
            LowRankRNNCell(hidden_size, hidden_size, rank)
            for _ in range(num_mechanisms)
        ])

        # Create input connection M*D*H
        self.W_in = nn.Parameter(0.01 * torch.randn(num_mechanisms, input_size, hidden_size))
        
        # Then create communication attention weights  # this is only for one head! M*H*K
        self.W_Q_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, key_size))
        self.W_K_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, key_size))
        self.W_V_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size * num_mechanisms, output_size)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_mechanisms, self.hidden_size, device=device)
    
    def rim_step(self, x, hidden_states=None,silence_attention=False):

        X = x # batch x input
        H = hidden_states # batch x mechanism x hidden

        # Input : X ( batch x input) * W_in ( mechanism x input x hidden)
        # Result: batch x mechanism x hidden
        A_in = torch.einsum('bi,mih->bmh', X, self.W_in)

        # recurrent update for all or only selected mechanisms
        H_in = H.clone()
        for i in range(self.num_mechanisms):
            h_in = self.mechanisms[i](A_in[:, i], H[:, i])
            H_in[:, i] = h_in

        # communicate between mechanism, only for active mechanism

        # Key: H (batch x mechanism x hidden) * W_K_comm (mechanism x hidden x key)
        # Result: batch x mechanism x key
        K_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_K_comm)

        # Query: H (batch x mechanism x hidden) * W_Q_in (mechanism x hidden x key)
        # Result: batch  x mechanism x key
        Q_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_Q_comm)

        # Value: H (batch x mechanism x hidden) * W_V_comm (mechanism x hidden x hidden)
        # Result: batch x mechanism x hidden
        V_comm = torch.einsum('bmh,mhh->bhm', H_in, self.W_V_comm)

        H_comm = H.clone() # should be H_in  
        
        # Then get attention weights across elements: softmax(Q * K^T / sqrt(d))
        # Result: batch x mechanisms x mechanisms (how much attention for each element)
        attention_com = torch.nn.Softmax(dim=-1)(
            torch.einsum('bmk,bnk->bmn', Q_comm, K_comm)
            / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float).to(device)))
        
        # Finally, get attention output by multiplying weights by values
        # Result: batch x mechanism x hidden
        A_comm = torch.einsum('bmn,bhm->bnh', attention_com, V_comm)

        if silence_attention :
            H_comm = H_in
        else :
            H_comm = H_in + A_comm
        
        return H_comm, A_in, A_comm, attention_com #  attn


    def forward(self, x, hidden_states=None,silence_attention=False):
        """
        Args:
            x: [batch_size, seq_length, input_size]
            hidden_states: optional initial hidden states
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        outputs = []
        attention_in = []
        attention_comm = []
        attention_comm_rim = []
        hidden_statess = []
        
        for t in range(seq_len):
            hidden_states, attn_input, attn_comm, attn_comm_rim = self.rim_step(x[:, t], hidden_states,silence_attention)
            
            combined_hidden = hidden_states.reshape(batch_size, -1)

            output = self.output_layer(combined_hidden)
            
            outputs.append(output)
            attention_in.append(attn_input)
            attention_comm.append(attn_comm)
            attention_comm_rim.append(attn_comm_rim)
            hidden_statess.append(hidden_states)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden_statess, attention_in, attention_comm, attention_comm_rim
    

class RIMNetwork_specialized(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_mechanisms=6,
                 key_size=16, rank=16, num_heads=4, key_size_input=32,task_size =0):
        super().__init__()
        
        assert num_mechanisms % 3 == 0, "num_mechanisms must be divisible by 3"
        
        self.input_size = input_size
        self.task_size = task_size
        self.sensory_size = input_size - task_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mechanisms = num_mechanisms
        self.mech_per_group = num_mechanisms // 3
        self.key_size = key_size
        self.rank = rank
        self.num_heads = num_heads

        # Create independent low-rank recurrent networks
        self.mechanisms = nn.ModuleList([
            LowRankRNNCell(hidden_size, hidden_size, rank)
            for _ in range(num_mechanisms)
        ])

        # Create specialized input connections
        # Task mechanisms (first third) only get task input
        self.W_in_task = nn.Parameter(0.01 * torch.randn(self.mech_per_group, task_size, hidden_size))
        # Sensory mechanisms (second third) only get sensory input
        self.W_in_sensory = nn.Parameter(0.01 * torch.randn(self.mech_per_group, self.sensory_size, hidden_size))
        # Output mechanisms (last third) get no direct input
        
        # Communication attention weights remain the same for all mechanisms
        self.W_Q_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, key_size))
        self.W_K_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, key_size))
        self.W_V_comm = nn.Parameter(0.01 * torch.randn(num_mechanisms, hidden_size, hidden_size))

        # Output layer only uses the last third of mechanisms
        self.output_layer = nn.Linear(hidden_size * self.mech_per_group, output_size)
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_mechanisms, self.hidden_size, device=device)
    
    def rim_step(self, x, hidden_states=None, silence_attention=False):
        X = x  # batch x input
        H = hidden_states  # batch x mechanism x hidden

        # Split input into task and sensory information
        X_task = X[:, -self.task_size:]  # batch x task_size
        X_sensory = X[:, :-self.task_size]  # batch x sensory_size

        # Initialize input processing tensor
        A_in = torch.zeros(X.size(0), self.num_mechanisms, self.hidden_size, device=X.device)

        # Process task input for first third of mechanisms
        A_in[:, :self.mech_per_group] = torch.einsum('bi,mih->bmh', X_task, self.W_in_task)
        
        # Process sensory input for second third of mechanisms
        A_in[:, self.mech_per_group:2*self.mech_per_group] = torch.einsum('bi,mih->bmh', X_sensory, self.W_in_sensory)
        
        # Last third receives no direct input (remains zero)

        # Recurrent update for all mechanisms
        H_in = H.clone()
        for i in range(self.num_mechanisms):
            h_in = self.mechanisms[i](A_in[:, i], H[:, i])
            H_in[:, i] = h_in

        # Communication between mechanisms (same as original)
        K_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_K_comm)
        V_comm = torch.einsum('bmh,mhh->bhm', H_in, self.W_V_comm)
        H_comm = H.clone()
        Q_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_Q_comm)

        attention_com = torch.nn.Softmax(dim=-1)(
            torch.einsum('bmk,bnk->bmn', Q_comm, K_comm)
            / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float).to(x.device)))
        
        A_comm = torch.einsum('bmn,bhm->bnh', attention_com, V_comm)

        if silence_attention:
            H_comm = H_in
        else:
            H_comm = H_in + A_comm
        
        return H_comm, A_in, A_comm, attention_com

    def forward(self, x, hidden_states=None, silence_attention=False):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        outputs = []
        attention_in = []
        attention_comm = []
        attention_comm_rim = []
        hidden_statess = []
        
        for t in range(seq_len):
            hidden_states, attn_input, attn_comm, attn_comm_rim = self.rim_step(x[:, t], hidden_states, silence_attention)
            
            # Only use the last third of mechanisms for output
            output_hidden = hidden_states[:, -self.mech_per_group:].reshape(batch_size, -1)
            output = self.output_layer(output_hidden)
            
            outputs.append(output)
            attention_in.append(attn_input)
            attention_comm.append(attn_comm)
            attention_comm_rim.append(attn_comm_rim)
            hidden_statess.append(hidden_states)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden_statess, attention_in, attention_comm, attention_comm_rim
    
    


class RIMNetwork_multHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_mechanisms=4,
                 key_size=16, rank=16, num_heads=1, task_size=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.num_mechanisms = num_mechanisms
        self.key_size = key_size
        self.rank = rank
        self.num_heads = num_heads

        # Create independent low-rank recurrent networks
        self.mechanisms = nn.ModuleList([
            LowRankRNNCell(hidden_size, hidden_size, rank)
            for _ in range(num_mechanisms)
        ])

        # Create input connection M*D*H
        self.W_in = nn.Parameter(0.01 * torch.randn(num_mechanisms, input_size, hidden_size))
        
        # Create attention weights for each head
        # Shape: H*M*D*K for Q and K, H*M*D*(D/H) for V where H is num_heads
        # Note: We divide hidden_size by num_heads for the value projection
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.W_Q_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, key_size))
        self.W_K_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, key_size))
        self.W_V_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, self.head_dim))
        
        # Linear transformation to combine head outputs
        # self.W_O = nn.Parameter(0.01 * torch.randn(num_mechanisms, num_heads * self.head_dim, hidden_size))

        self.output_layer = nn.Linear(hidden_size * num_mechanisms, output_size)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_mechanisms, self.hidden_size, device=device)
    
    def rim_step(self, x, hidden_states=None, silence_attention=False, smooth_attention_weights = None):
        X = x  # batch x input
        H = hidden_states  # batch x mechanism x hidden

        # Input processing
        A_in = torch.einsum('bi,mih->bmh', X, self.W_in)

        # Recurrent update
        H_in = H.clone()
        for i in range(self.num_mechanisms):
            h_in = self.mechanisms[i](A_in[:, i], H[:, i])
            H_in[:, i] = h_in

        if silence_attention:
            return H_in, A_in, None, None

        # Multi-head attention computation
        A_comm_heads = []
        attention_weights_heads = []
        
        for head in range(self.num_heads):
            # Compute K, Q, V for current head
            K_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_K_comm[head])
            Q_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_Q_comm[head])
            V_comm = torch.einsum('bmh,mhh->bmh', H_in, self.W_V_comm[head])

            # Compute attention weights
            attention_weights = torch.nn.Sigmoid()(                                                 # Softmax     or softplus?        
                torch.einsum('bmk,bnk->bmn', Q_comm, K_comm)
                / ( torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32).to(x.device)))
            )

            if smooth_attention_weights is not None:
                attention_weights = 0.5 * attention_weights + 0.5 * smooth_attention_weights[head]
            
            # Compute attention output for this head
            A_comm_head = torch.einsum('bmn,bnh->bmh', attention_weights, V_comm)
            
            A_comm_heads.append(A_comm_head)
            attention_weights_heads.append(attention_weights)
        
        # Sum attention outputs from all heads
        A_comm = torch.stack(A_comm_heads).sum(dim=0)

        attention_weights_heads = torch.stack(attention_weights_heads)
        
        # Update hidden states with attention output +> wouldn't it make sense that it is the average of both? So that 
        # when the main attention of a mechanism is on itself, the attention just act as the identity? It seems better somehow

        # if A_comm_smooth is not None:
        #     A_comm = 0.5 * A_comm + 0.5 * A_comm_smooth

        H_comm = H_in + A_comm
        
        return H_comm, A_in, A_comm, attention_weights_heads

    def forward(self, x, hidden_states=None, silence_attention=False):
        """
        Args:
            x: [batch_size, seq_length, input_size]
            hidden_states: optional initial hidden states
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        outputs = []
        attention_in = []
        attention_comm = []
        attention_weights = []
        hidden_states_list = []
        
        attn_weights = None

        for t in range(seq_len):
            hidden_states, attn_input, attn_comm, attn_weights = self.rim_step(
                x[:, t], hidden_states, silence_attention,smooth_attention_weights=attn_weights
            )
            
            combined_hidden = hidden_states.reshape(batch_size, -1)
            output = self.output_layer(combined_hidden)
            
            outputs.append(output)
            attention_in.append(attn_input)
            if attn_comm is not None:
                attention_comm.append(attn_comm)
                attention_weights.append(attn_weights)
            hidden_states_list.append(hidden_states)
        
        outputs = torch.stack(outputs, dim=1)
        
        if silence_attention:
            attention_comm = None
            attention_weights = None
        else:
            attention_comm = torch.stack(attention_comm, dim=1)
            attention_weights = torch.stack(attention_weights, dim=1)
            
        return outputs, hidden_states_list, attention_in, attention_comm, attention_weights
    









class RIMNetwork_specialized_multHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_mechanisms=6,
                 key_size=16, rank=16, num_heads=4, key_size_input=32, task_size=0):
        super().__init__()
        
        assert num_mechanisms % 3 == 0, "num_mechanisms must be divisible by 3"
        
        self.input_size = input_size
        self.task_size = task_size
        self.sensory_size = input_size - task_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mechanisms = num_mechanisms
        self.mech_per_group = num_mechanisms // 3
        self.key_size = key_size
        self.rank = rank
        self.num_heads = num_heads

        # Create independent low-rank recurrent networks
        self.mechanisms = nn.ModuleList([
            LowRankRNNCell(hidden_size, hidden_size, rank)
            for _ in range(num_mechanisms)
        ])

        # Create specialized input connections
        # Task mechanisms (first third) only get task input
        self.W_in_task = nn.Parameter(0.01 * torch.randn(self.mech_per_group, task_size, hidden_size))
        # Sensory mechanisms (second third) only get sensory input
        self.W_in_sensory = nn.Parameter(0.01 * torch.randn(self.mech_per_group, self.sensory_size, hidden_size))
        # Output mechanisms (last third) get no direct input
        
        # Create multi-head attention weights
        # Shape: H*M*D*K for Q and K, H*M*D*D for V where H is num_heads
        self.W_Q_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, key_size))
        self.W_K_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, key_size))
        self.W_V_comm = nn.Parameter(0.01 * torch.randn(num_heads, num_mechanisms, hidden_size, hidden_size))

        # Output layer only uses the last third of mechanisms
        self.output_layer = nn.Linear(hidden_size * self.mech_per_group, output_size)
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.num_mechanisms, self.hidden_size, device=device)
    
    def rim_step(self, x, hidden_states=None, silence_attention=False,smooth_attention_weights=None):
        X = x  # batch x input
        H = hidden_states  # batch x mechanism x hidden

        # Split input into task and sensory information
        X_task = X[:, -self.task_size:]  # batch x task_size
        X_sensory = X[:, :-self.task_size]  # batch x sensory_size

        # Initialize input processing tensor
        A_in = torch.zeros(X.size(0), self.num_mechanisms, self.hidden_size, device=X.device)

        # Process task input for first third of mechanisms
        A_in[:, :self.mech_per_group] = torch.einsum('bi,mih->bmh', X_task, self.W_in_task)
        
        # Process sensory input for second third of mechanisms
        A_in[:, self.mech_per_group:2*self.mech_per_group] = torch.einsum('bi,mih->bmh', X_sensory, self.W_in_sensory)
        
        # Last third receives no direct input (remains zero)

        # Recurrent update for all mechanisms
        H_in = H.clone()
        for i in range(self.num_mechanisms):
            h_in = self.mechanisms[i](A_in[:, i], H[:, i])
            H_in[:, i] = h_in

        if silence_attention:
            return H_in, A_in, None, None

        # Multi-head attention computation
        A_comm_heads = []
        attention_weights_heads = []
        
        for head in range(self.num_heads):
            # Compute K, Q, V for current head
            K_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_K_comm[head])
            Q_comm = torch.einsum('bmh,mhk->bmk', H_in, self.W_Q_comm[head])
            V_comm = torch.einsum('bmh,mhh->bmh', H_in, self.W_V_comm[head])

            # Compute attention weights
            attention_weights = torch.nn.Sigmoid()(
                torch.einsum('bmk,bnk->bmn', Q_comm, K_comm)
                / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32).to(x.device))
            )

            if smooth_attention_weights is not None:
                attention_weights = 0.5 * attention_weights + 0.5 * smooth_attention_weights[head]

            # Compute attention output for this head
            A_comm_head = torch.einsum('bmn,bnh->bmh', attention_weights, V_comm)
            
            A_comm_heads.append(A_comm_head)
            attention_weights_heads.append(attention_weights)
        
        # Sum attention outputs from all heads
        A_comm = torch.stack(A_comm_heads).sum(dim=0)
        attention_weights_heads = torch.stack(attention_weights_heads)
        
        # Update hidden states with attention output
        H_comm = H_in + A_comm
        
        return H_comm, A_in, A_comm, attention_weights_heads

    def forward(self, x, hidden_states=None, silence_attention=False):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device)
        
        outputs = []
        attention_in = []
        attention_comm = []
        attention_weights = []
        hidden_states_list = []
        
        attn_weights = None
        for t in range(seq_len):
            hidden_states, attn_input, attn_comm, attn_weights = self.rim_step(
                x[:, t], hidden_states, silence_attention,smooth_attention_weights=attn_weights
            )
            
            # Only use the last third of mechanisms for output
            output_hidden = hidden_states[:, -self.mech_per_group:].reshape(batch_size, -1)
            output = self.output_layer(output_hidden)
            
            outputs.append(output)
            attention_in.append(attn_input)
            if attn_comm is not None:
                attention_comm.append(attn_comm)
                attention_weights.append(attn_weights)
            hidden_states_list.append(hidden_states)
        
        outputs = torch.stack(outputs, dim=1)
        
        if silence_attention:
            attention_comm = None
            attention_weights = None
        else:
            attention_comm = torch.stack(attention_comm, dim=1)
            attention_weights = torch.stack(attention_weights, dim=1)
            
        return outputs, hidden_states_list, attention_in, attention_comm, attention_weights