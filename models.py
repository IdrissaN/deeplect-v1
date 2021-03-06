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
from attention import *
from transformers import BertModel, BertConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model_features = BertModel.from_pretrained('bert-base-uncased', config=config)
for param in model_features.parameters():
    param.requires_grad = False
    
    
    
def dim_calcul_conv2d(N, C_in, H_in, W_in, layer):

    padding = layer.padding
    stride = layer.stride
    dilation = layer.dilation
    kernel_size = layer.kernel_size
    C_out = layer.out_channels
    H_out = 1 + (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]
    W_out = 1 + (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]
        
    return (N, C_out, H_out, W_out)


def dim_calcul_avg_pool2d(N, C_in, H_in, W_in, layer):

    padding = layer.padding
    kernel_size = layer.kernel_size
    stride = layer.stride
    C_out = C_in
    H_out = 1 + (H_in + 2 * padding[0] - kernel_size[0]) // stride[0]
    W_out = 1 + (W_in + 2 * padding[1] - kernel_size[1]) // stride[1]
        
    return (N, C_out, H_out, W_out)


def dim_calcul_max_pool2d(N, C_in, H_in, W_in, layer):

    padding = layer.padding
    kernel_size = layer.kernel_size
    stride = layer.stride
    dilation = layer.dilation
    C_out = C_in
    H_out = 1 + (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1)- 1) // stride[0]
    W_out = 1 + (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1)- 1) // stride[1]
        
    return (N, C_out, H_out, W_out)
    
    
class ConvBase(nn.Module):
    def __init__(self, hidden_size, batch_sz, spec_dim):
        super().__init__() 
        self.hidden_size = hidden_size
        self.pool_1 = nn.AvgPool2d(kernel_size=(3,3), stride=(2, 3), padding=(0,0))
        #self.pool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 3), padding=(0,0), dilation=(1,1))
        self.batchnorm2d_1 = nn.BatchNorm2d(hidden_size)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=1, dilation=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, dilation=1)
        # Compute the dimension that we will give to  the GRU
        output_dim_1 = dim_calcul_conv2d(batch_sz, *spec_dim, self.conv2d_1)
        output_dim_2 = dim_calcul_conv2d(*output_dim_1, self.conv2d_2)
        output_dim_3 = dim_calcul_avg_pool2d(*output_dim_2, self.pool_1)
        #output_dim_3 = dim_calcul_max_pool2d(*output_dim_2, self.pool_1)
        self.input_gru_dim = output_dim_3[1] * output_dim_3[2]
        self.encoder_timestamp = output_dim_3[-1]

    
    def forward(self, mfccs):
        # Started with the convolutionnal base
        conv_layer = self.conv2d_1(mfccs)
        conv_layer = self.conv2d_2(F.relu(conv_layer))
        conv_layer = self.batchnorm2d_1(F.relu(conv_layer))
        avg_layer = self.pool_1(conv_layer) 
        # Reshape from (batch, channels, features, time) ----> (batch, time, channels * features)
        dim_0, dim_1, dim_2, dim_3 = avg_layer.shape
        output = avg_layer.reshape(dim_0, dim_3, dim_1 * dim_2)
        return output
    
    
class RnnBase(nn.Module):
    def __init__(self, device, hidden_size, batch_sz, bn_in_feat, gru_in_feat):
        super().__init__() 
        self.device = device
        self.hidden_size = hidden_size
        self.batch_sz = batch_sz 
        self.gru = nn.GRU(gru_in_feat, self.hidden_size, batch_first=True, num_layers=1, bidirectional=True, dropout=0.2)
    
    
    def forward(self, seq, hidden):
        #hidden = self.initialize_hidden_state()
        # Beginning rnn bloc
        output, hidden = self.gru(seq, hidden)
        _, dim_1, dim_2 = output.shape
        # Separate the forward pass ----> batch, seq_len, num_directions, hidden_size
        forward = output.view(-1, dim_1, 2, self.hidden_size)[:, :, 0, :]
        # Separate the backward pass  
        backward = output.view(-1, dim_1, 2, self.hidden_size)[:, :, 1, :]
        # Sum the forward pass and the backward to form the output
        output = forward + backward
        output = F.relu(output)
        # Pass through the dropout layer
        # I don't know what i will do with but i collect the last state for the moment
        hidden = hidden.view(1, 2,  -1, self.hidden_size)
        h_forward = hidden[:, 0, :, :]
        h_backward = hidden[:, 1, :, :]
        # Type of merge chosen 
        hidden = h_forward + h_backward # Batch norm pour le hidden ?
        
        return output, hidden

        
    
class EncoderCONV2DRNN(nn.Module):
    def __init__(self, device, batch_sz, hidden_size, spec_dim):
        super().__init__() 
        
        self.batch_sz = batch_sz
        self.conv_base = ConvBase(hidden_size, batch_sz, spec_dim=spec_dim)
        self.device = device
        self.hidden_size = hidden_size
        self.encoder_timestamp = self.conv_base.encoder_timestamp
        self.norm_layer_1 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        self.norm_layer_2 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        self.norm_layer_3 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        self.norm_layer_4 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        self.norm_layer_5 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        self.norm_layer_6 = nn.LayerNorm((self.encoder_timestamp, hidden_size))
        
        self.rnn_base_1 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=self.conv_base.input_gru_dim)
        self.rnn_base_2 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=hidden_size)
        self.rnn_base_3 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=hidden_size)
        self.rnn_base_4 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=hidden_size)
        self.rnn_base_5 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=hidden_size)
        self.rnn_base_6 = RnnBase(device, hidden_size, batch_sz, bn_in_feat=self.encoder_timestamp, gru_in_feat=hidden_size)


    def forward(self, mfccs, hidden):
        
        # Convolutionnal base
        output = self.conv_base(mfccs)
        # Sequential bloc
        #1
        output, _ = self.rnn_base_1(output, hidden)
        output = self.norm_layer_1(output)
        copy_output = output.clone()
        #2
        output, _ = self.rnn_base_2(output, hidden)
        output = output + copy_output
        output = self.norm_layer_2(output)
        copy_output = output.clone()
        #3
        output, _ = self.rnn_base_3(output, hidden)
        output = output + copy_output
        output = self.norm_layer_3(output)
        copy_output = output.clone()
        #4
        output, _ = self.rnn_base_4(output, hidden)
        output = output + copy_output
        output = self.norm_layer_4(output)
        copy_output = output.clone()
        #5
        output, _ = self.rnn_base_5(output, hidden)
        output = output + copy_output
        output = self.norm_layer_5(output)
        copy_output = output.clone()
        #6
        output, hidden = self.rnn_base_6(output, hidden)
        output = output + copy_output
        output = self.norm_layer_6(output)

        return output, hidden

    def initialize_hidden_state(self):
        return torch.zeros(2, self.batch_sz, self.hidden_size, device=self.device)
    


class bert_embedding_layer(nn.Module):
    def __init__(self, model=model_features):
        super().__init__()
        # Load pre-trained model (weights)
        self.model = model
        self.model.eval()
        self.model.config.output_hidden_states=True

        
    def forward(self, tokens_tensor):

        segments_tensor = torch.ones(tokens_tensor.shape[0], tokens_tensor.shape[1], device=device)

        last_layers, _, encoded_layers = self.model(tokens_tensor, segments_tensor)
            
        return torch.cat(encoded_layers[-4:], dim=-1)
    
    

class DecoderATTRNN(nn.Module):
    """Bahdanau Audio decoder"""
    
    def __init__(self, vocab_size, dec_units, batch_sz, hidden_size, encoder_timestamp=198):
        
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = bert_embedding_layer()
        
        self.norm_layer_1 = nn.LayerNorm((1, hidden_size))
        self.norm_layer_2 = nn.LayerNorm((1, hidden_size))
        self.norm_layer_3 = nn.LayerNorm((1, hidden_size))
        
        self.gru_1 = nn.GRU(3072 + hidden_size, self.dec_units, batch_first=True, dropout=0.1)
        self.gru_2 = nn.GRU(self.dec_units, self.dec_units, batch_first=True, dropout=0.1)
        self.gru_3 = nn.GRU(self.dec_units, self.dec_units, batch_first=True, dropout=0.1)
        self.gru_4 = nn.GRU(self.dec_units, self.dec_units, batch_first=True, dropout=0.1)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        # used for attention
        #self.attention = BahdanauAttentionAudio(units=dec_units, hidden_size=hidden_size)
        self.attention =  AverageHeadAttention(dec_units, hidden_size, encoder_timestamp)
 
    
    def forward(self, input, hidden, enc_output, prev_attn_weights):
        # enc_output shape == (batch_sz, max_length, hidden_size)
        context_vector, attention_weights, _ = self.attention(hidden, enc_output, prev_attn_weights)
        # x shape after passing input through embedding == (batch_sz, 1, embedding_dim)
        x = self.embedding(input)
        # x shape after concatenation == (batch_sz, 1, embedding_dim + hidden_size)
        context_vector = torch.unsqueeze(context_vector, 1)
        x = torch.cat((context_vector, x), 2)
        # passing the concatenated vector to the GRU
        #1
        output, _ = self.gru_1(x)
        copy_output = output.clone()
        #2
        output, _ = self.gru_2(output)
        output = output + copy_output
        output = self.norm_layer_1(output)
        copy_output = output.clone()
        #3
        output, _ = self.gru_3(output)
        output = output + copy_output
        output = self.norm_layer_2(output)
        copy_output = output.clone()
        #4
        output, state = self.gru_4(output)
        output = output + copy_output
        output = self.norm_layer_3(output)
        # output shape == (batch_sz * 1, hidden_size)
        output = output.reshape(-1, output.shape[2])
        # output shape == (batch_sz, vocab)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)

        return output, state, attention_weights
        