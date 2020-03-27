            #########################################################
            #                                                       #
            #  This code is the private poperty of dialectai. It's  #
            #  forbidden to copy it, use it or sell it outside of   # 
            #  the company. Copyright - dialectai 2020.             #
            #                                                       #
            #########################################################



import torch
import torch.nn as nn
import torch.nn.functional as F




def smoothing(x):
    """Compute softmax values for each sets of scores in x.
    By changing the exponential by a sigmoid"""
    return x.sigmoid() / x.sigmoid().sum(0)



        
class BahdanauAttentionAudio(nn.Module):
    
    def __init__(self, units, hidden_size, encoder_timestamps=198):
        super().__init__()
        self.kernel_size = 3
        self.encoder_timestamps = encoder_timestamps
        self.W1 = nn.Linear(hidden_size, units)
        self.W2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)
        self.loc_conv = nn.Conv1d(in_channels=encoder_timestamps, out_channels=encoder_timestamps, 
                                  kernel_size=2*self.kernel_size+1, padding=self.kernel_size, bias=False)
        self.loc_proj = nn.Linear(1, hidden_size, bias=False)
        
    def forward(self, query, values, prev_att):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query = torch.squeeze(query, 0)
        hidden_with_time_axis = torch.unsqueeze(query, 1)
        # Calculate location context
        convo = self.loc_conv(prev_att)
        loc_context = self.loc_proj(convo)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        sum_1 = self.W1(values) + self.W2(hidden_with_time_axis) + loc_context
        score = self.V(torch.tanh(sum_1))
        # As we are dealing with audio we will take the topk frames
        top_val, top_pos = torch.topk(score, k=(self.encoder_timestamps * 2) // 3, dim=1)
        score = score.squeeze(2)
        top_pos = top_pos.squeeze(2)
        score = -1 * (score.scatter(1, top_pos, 0) - score)
        score = score.unsqueeze(2)
        # We will change the softmax with the smoothing
        attention_weights = smoothing(score)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score 
                      
                      

        

class SuperHeadAttention(nn.Module):
    
    def __init__(self, units, hidden_size, encoder_timestamps=198):
        super().__init__()
        self.nbr_heads = 3
        self.encoder_timestamps = encoder_timestamps
        self.W = nn.Linear(self.nbr_heads, 1)
        self.attention_1 = BahdanauAttentionAudio(units=units, hidden_size=hidden_size, encoder_timestamps=encoder_timestamps)
        self.attention_2 = BahdanauAttentionAudio(units=units, hidden_size=hidden_size, encoder_timestamps=encoder_timestamps)
        self.attention_3 = BahdanauAttentionAudio(units=units, hidden_size=hidden_size, encoder_timestamps=encoder_timestamps)

        
    def forward(self, query, values, prev_att):
        
        _, _, score_1 = self.attention_1(query, values, prev_att)
        _, _, score_2 = self.attention_2(query, values, prev_att)
        _, _, score_3 = self.attention_3(query, values, prev_att)
        
        concat = torch.cat([score_1, score_2, score_3], dim=2)
        score = self.W(concat)
        # As we are dealing with audio we will take the topk frames
        top_val, top_pos = torch.topk(score, k=(self.encoder_timestamps * 2) // 3, dim=1)
        score = score.squeeze(2)
        top_pos = top_pos.squeeze(2)
        score = -1 * (score.scatter(1, top_pos, 0) - score)
        score = score.unsqueeze(2)
        # We will change the softmax with the smothing    
        attention_weights = smoothing(score)
        #attention_weights = F.softmax(score, dim=1)
        # context_vector shape after sum == (batch_size, hidden_size) 
        # (values == EO)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights, score
        
        
        