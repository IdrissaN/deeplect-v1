            #########################################################
            #                                                       #
            #  This code is the private poperty of dialectai. It's  #
            #  forbidden to copy it, use it or sell it outside of   # 
            #  the company. Copyright - dialectai 2020.             #
            #                                                       #
            #########################################################


            
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
import importlib
import tokenizers
import unicodedata
import numpy as np
import tensorflow as tf
import torchaudio
from queue import PriorityQueue
from torch.utils import data
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from SpecAugment import spec_augment_pytorch
from nltk.translate.bleu_score import sentence_bleu as bleu
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files)


warnings.filterwarnings('ignore')



# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn')



def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9_*?.!,¿?åäöèÅÄÖÈÉçëË]+<>", " ", w)

    w = w.rstrip().strip()
    w = "[CLS] " + w + " [SEP]"
        
    return w


def load_librispeech_item(fileid, path, ext_audio, ext_txt, text_only=False):
    
    speaker_id, chapter_id, utterance_id = fileid.split("-")
    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)
    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    # Load audio
    if text_only is False:
        wav, sr = torchaudio.load(file_audio)
        wav = wav.numpy()[0]     
    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text: 
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)
    if text_only is False:
        return wav, sr, utterance
    else:
        return utterance
    

class LibriSpeechDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"
    
    
    def __init__(self, limit=None, n_channels=1, n_frames=128, sr=16000, n_fft=2048, max_target_length=40,
                 n_mels=39, hop_length=512, power=1.0, n_mfcc=39, duration=10, path="../librispeech/LibriSpeech/",
                 version = "train-clean-360", method='mel'):
        'Initialization'
        self.method = method
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Bert's tokenizer
        self.limit = limit 
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.power = power
        self.n_mfcc = n_mfcc
        self.duration = duration
        self._path = os.path.join(path, version)
        walker = walk_files(self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True)
        self._walker = list(walker)[:limit]
        self.max_length = max_target_length
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._walker)
    
    
    
    def spec_augmenter(self, spec):
        """ Perform spectogram level's augmentation for audio data."""
        spec = spec_augment_pytorch.spec_augment(mel_spectrogram=spec)
        return spec
    

    def background_noise(self, wave):
        """
        """
        noise_factor = np.random.choice([-0.1, 0, 0.1])
        noise = np.random.randn(len(wave))
        augmented_wave = wave + noise_factor * noise
        # Cast back to same data type
        augmented_wave = augmented_wave.astype(type(wave[0]))

        return augmented_wave
         
    
    def change_speed(self, wave):
        """
        """
        speed_factor = np.random.choice([0.9, 1, 1.1])
        augmented_wave = librosa.effects.time_stretch(wave, speed_factor)

        return augmented_wave
    
    

    def choice_augmentation(self, wave):
        """Choose an augmentation technique from between adding noise or changing speed.
        """
        yes_or_not = np.random.choice([False, True], p=[0.65, 0.35])
        if yes_or_not == True:
            # Choose randomly an augmentation
            aug = np.random.choice(["b_noise", "speed"], p=[0.30, 0.70])
            #  and perform the augmentation 
            if aug == "b_noise":
                wave = self.background_noise(wave)
            elif aug == "speed":
                wave = self.change_speed(wave)

            return wave
        else:
            return wave

        

    def wave_augmenter(self, wave):
        """ Choose to randomly apply an augmentation to a wave sequence."""
        # We will perform randomly an augmentation here
        wave = self.choice_augmentation(wave)
        # Define the minimum length of the wave 
        length = self.duration * self.sr
        if len(wave) < self.sr:
            wave = np.array(np.pad(wave, (0, length - len(wave)), 'constant', constant_values= 0))
        else:
            wave = wave[:length]

        return wave
    
    
    def wave2mfcc(self, wave):
        """ 
        Opens an wav audio file with librosa and converts it in mfccs features.
        
        :param path_wav:
        
        """
        # We create the mfcc
        mfccs = librosa.feature.mfcc(y=wave, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                     power=self.power, n_mels=self.n_mels, n_mfcc=self.n_mfcc)
        # Normalization of the mfccs
        mfccs = ((mfccs.T - mfccs.mean(axis=1)) / mfccs.std(axis=1)).T 
        
        return mfccs 
    
    
    def wave2melspec(self, wave):
        mel_spectogram =  librosa.feature.melspectrogram(y=wave, sr=self.sr, n_mels=self.n_mels,
                                                         hop_length=self.hop_length, fmax=self.sr)
        
        return mel_spectogram
    
    
    def str2num(self, sentence, lang_tokenizer):

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = lang_tokenizer.encode(sentence, add_special_tokens=True) 

        return indexed_tokens
    
    
    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        fileid = self._walker[index]
        # Load data and get label
        wav, sr, sentence = load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)
        # Random augmentation at wav level
        wav = self.wave_augmenter(wav)
        # transformer en spec
        if self.method == 'mfccs':
            mat = self.wave2mfcc(wav)
        elif self.method == 'mel':
            mat = self.wave2melspec(wav)

        # Padding to have the same number of frame in each mfccs
        if mat.shape[1] < self.n_frames:
            mat = np.array(np.pad(mat, ((0,0), (0, self.n_frames - mat.shape[1])), 'constant', constant_values=0)) 
        else:
            mat = mat[:, :self.n_frames]
        # 
        mat = mat.reshape((1, *mat.shape))
        sentence = preprocess_sentence(sentence)
        sentence = self.str2num(sentence, self.tokenizer)#[0]
        if len(sentence) < self.max_length:
            sentence = np.array(np.pad(sentence, (0, self.max_length - len(sentence)), 'constant', constant_values=0)) 
        else:
            sentence = sentence[:self.max_length]     
        mat = torch.tensor(mat)
        # Random augmentation at sepectogram level for mel spectogram
        #if self.method == 'mel':
        mat = self.spec_augmenter(mat)
        
        return mat, torch.tensor(sentence)
    
    

    
    
def train_step(phase, input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device, batch_sz, targ_lang_tokenizer, teacher_forcing_ratio=0.80):
    
    # Initialize the encoder
    encoder_hidden = encoder.initialize_hidden_state()
    # Put all the previously computed gradients to zero
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(1)
    # Encode the input sentence
    decoder_attention = torch.zeros(batch_sz, 198, 1).to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    
    loss = 0
    decoder_input = torch.tensor([[targ_lang_tokenizer.cls_token_id]] * batch_sz, device=device)
    decoder_hidden =  encoder_hidden
    # Use randomly teacher forcing
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True  
    else:  
        use_teacher_forcing = False

    #if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input to help the model
    # in case it starts with the wrong word.
    for di in range(1, target_length):
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                    encoder_outputs, decoder_attention)
           
            
        loss += criterion(decoder_output, target_tensor[:, di])
        if use_teacher_forcing:
            decoder_input = torch.unsqueeze(target_tensor[:, di], 1)  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.data.topk(1)
            # the predicted ID is fed back into the model
            decoder_input = topi.detach()

    batch_loss = (loss.item() / int(target_tensor.shape[1]))
    if phase == 'train':
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return batch_loss



def global_trainer(nbr_epochs, train_dataloader, valid_dataloader, encoder, decoder, 
                   encoder_optimizer, decoder_optimizer,criterion, device, batch_sz, 
                   tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                   patience=10):
    
    nb_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(" ======"*6)
    print("      The model has {} parameters".format(nb_params))
    print(" ======"*6)
    print("\n")
    
    start = time.time()
    best_val_loss = np.inf
    count = 0
    for epoch in range(nbr_epochs):
        with tqdm.tqdm(total=len(train_dataloader), file=sys.stdout, leave=True, desc='Epoch ', \
                       bar_format="{l_bar}{bar:20}{r_bar}{bar:-15b}") as pbar:  
            for phase in ['train', 'eval']:
                if phase == 'train':
                    encoder.train()
                    decoder.train()
                    dataloader = train_dataloader
                else:
                    encoder.eval()
                    decoder.eval()
                    dataloader = valid_dataloader
                    
                total_loss = 0
                for batch, (inp, targ) in enumerate(dataloader):
                    pbar.set_description('Epoch {:>8}'.format(epoch + 1))
                    inp, targ = inp.to(device), targ.to(device)
                    if phase == 'train':
                        batch_loss = train_step(phase, inp, targ, encoder, decoder, encoder_optimizer,
                                        decoder_optimizer, criterion,
                                        device, batch_sz, targ_lang_tokenizer=tokenizer)
                    else:
                        with torch.no_grad():
                            batch_loss = train_step(phase, inp, targ, encoder, decoder, encoder_optimizer,
                                        decoder_optimizer, criterion,
                                        device, batch_sz, targ_lang_tokenizer=tokenizer)
                    total_loss += batch_loss
                
                    if phase == 'train':
                        train_loss = total_loss / (batch + 1)
                        pbar.set_postfix_str('Train loss {:.4f}'.format(train_loss))
                        pbar.update(1)
                        time.sleep(1)
                    else:
                        val_loss = total_loss / (batch + 1)
                        pbar.set_postfix_str('Train loss {:.4f} Eval loss {:.4f}'.format(train_loss, val_loss))
                        time.sleep(1)  

        # saving (checkpoint) the model each epoch when we aren't overfitting
        
        if val_loss < best_val_loss:
  
            best_val_loss = val_loss
            torch.save(encoder, 'encoder-s2t.pt')
            torch.save(decoder, 'decoder-s2t.pt')
        else:
            count += 1
            if count == patience:
                print("\n")
                print(" ----- EARLY STOPPING ----- ") 
                break
        
        
            

    print('\nTime taken for the training {:.5} hours\n'.format((time.time() - start) / 3600))
        