import os
import re
import csv
import sys
import tqdm
import time
import torch
import random
import librosa
import warnings
import unicodedata
import numpy as np
import tensorflow as tf
import torchaudio
from queue import PriorityQueue
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from SpecAugment import spec_augment_pytorch
from nltk.translate.bleu_score import sentence_bleu as bleu
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files)

warnings.filterwarnings('ignore')


    
class BeamTreeNode(object):
    "Generic tree node."
        
    def __init__(self, name, hidden_state, wordid=1, logp=1,  children=None, parent=None):
        self.name = name
        self.h = hidden_state
        self.wordid = wordid
        self.logp = logp
        self.children = []
        self.parent = parent
        self.is_leaf = True
        self.is_end = False
        self.length = 0
        self.inv_path = [self.wordid.item()]

    @property
    def path(self):
        return self.inv_path[::-1]
    
    def __lt__(self, other):
        return self.logp.item() < other.logp.item() 
    
    def __eq__(self, other):
        return self.logp.item() == other.logp.item() 

    def __repr__(self):
        return self.name
    
    def is_child(self, node):
        return node in self.children
        
    def add_child(self, node):
        assert isinstance(node, BeamTreeNode)
        assert self.is_child(node) == False
        self.children.append(node)
        self.is_leaf = False

    def add_parent(self, node):
        assert isinstance(node, BeamTreeNode)
        self.parent = node
        self.length = node.length + 1
        self.inv_path += node.inv_path
    


        
def greedy_decode(mfccs, max_length_targ, encoder, decoder, targ_lang, device, enc_units=64, bad_audio_mode=True):

    
    # Send the inputs matrix to device
    mfccs = torch.tensor(mfccs).to(device)

    result = ''
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        enc_hidden = torch.zeros(2, 1, enc_units, device=device)
        enc_out, enc_hidden = encoder(mfccs, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word_index['<start>']]], device=device)
        attention_weights = torch.zeros(1, 198, 1).to(device)
        for t in range(max_length_targ):
            if bad_audio_mode:
                predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                     dec_hidden,
                                                                     enc_out,
                                                                     attention_weights)
            else:  
                predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                     dec_hidden,
                                                                     enc_out)
            # storing the attention weights to plot later on
            topv, topi = predictions.data.topk(1)
            result += targ_lang.index_word[topi.item()] + ' '

            if targ_lang.index_word[topi.item()] == '<end>':
                return result, sentence

            # the predicted ID is fed back into the model
            dec_input = torch.tensor([topi.item()], device=device).unsqueeze(0)

        return result
    
        
def beam_search_decode(mfccs, max_length_targ,  encoder, decoder,  targ_lang, device,
                       nb_candidates, beam_width, alpha, enc_units=64, bad_audio_mode=True):

    # Send the inputs matrix to device
    mfccs = torch.tensor(mfccs).to(device)

    result = ''

    with torch.no_grad():
        hidden = torch.zeros(2, 1, enc_units, device=device)
        attention_weights = torch.zeros(1, 198, 1).to(device)
        
        enc_out, enc_hidden = encoder(mfccs, hidden)

        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word_index['<start>']]], device=device)
        
        candidates = []
        # Créer la racine (le noeud de départ de l'arbre)
        node = BeamTreeNode(name='root', hidden_state=dec_hidden, wordid=dec_input, logp=torch.tensor(0, device=device))
        candidates.append(node)
        
        count = 0
        endnodes = []
        for t in range(max_length_targ):
            all_nodes = PriorityQueue()
            for n in candidates:
                if n.is_leaf and not n.is_end:
                    # étendre le noeud (faire les prédictions dessus)
                    if bad_audio_mode:
                        predictions, dec_hidden, attention_weights = decoder(n.wordid, n.h, enc_out,
                                                                             attention_weights)
                    else:
                        predictions, dec_hidden, attention_weights = decoder(n.wordid, n.h, enc_out)
                    # Pour signaler que le noeud est déjà étendu (utilisé)
                    n.is_leaf = False
                    # prendre le nombre de candidats choisis 
                    top_width_v, top_width_i = predictions.data.topk(nb_candidates)
                    # Créer beam width noeuds pour stocker les prédictions et les rajouter à 
                    # la liste de noeuds à scorer
                    for val, ind in zip(top_width_v[0], top_width_i[0]):
                        count += 1
                        dec_input = torch.tensor([[ind.item()]], device=device)
                        logproba = - val + n.logp
                        node = BeamTreeNode(name=str(count), hidden_state=dec_hidden, wordid=dec_input, logp=logproba)
                        # Rajouter le noeud à la priority queue
                        all_nodes.put(node)
                        # Indiquer que les nouveaux noeuds sont des enfants du noeud initial 
                        n.add_child(node)
                        node.add_parent(n)
                        # Si on prédit la fin ou que la longueur maximale est atteinte
                        if targ_lang.index_word[ind.item()] == '<end>':
                            node.is_end = True 
                            endnodes.append(node)
       
            # Retenir que les beam width meilleurs           
            candidates = [all_nodes.get() for step in range(beam_width)]
            #candidates = [node for _, node in candidates]
            candidates = [node for node in candidates]
            
        # Last step before the result 
        final_queue = PriorityQueue()
        final_candidates = candidates + endnodes
        # Put all final candidates nodes in a priority queue and choose the best one based 
        # on the score and not on the logp
        for n in final_candidates:
            score = n.logp / (n.length ** alpha)
            final_queue.put((score, n))
        # Choose the best node  
        _, node = final_queue.get()
        # Find the path
        for elem in node.path:
            if elem != 0:
                result += targ_lang.index_word[elem] + ' '

        return result        
        
        

def evaluate(mfccs, references, max_length_targ, encoder, decoder, targ_lang, 
              device, beam_search=False, beam_width=3, alpha=0.3, nb_candidates=10):
    
    if beam_search == False:
        result= greedy_decode(mfccs, max_length_targ, encoder, decoder, targ_lang, device)
    else:
        result = beam_search_decode(mfccs, max_length_targ, encoder, decoder, targ_lang, device=device,
                                              beam_width=beam_width, nb_candidates=nb_candidates, alpha=alpha)
    result = result.split()    
    BLEUscore = bleu([references], result, weights = (0.5, 0.5))
    
    print("Input: {}".format(references))
    print("\n")
    print("Predicted translation: {}".format(result))
    print("\n")
    print("Bleu score: {}".format(BLEUscore))
 