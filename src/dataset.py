import torch
import os
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import librosa


class HiveAudioDataset(Dataset):
    
    
    """Creating the Audio Dataset"""
    
    def __init__(self,file_path, labels , sample_rate = 22050, duration= 3.0 , n_mels = 128):
        self.file_path = file_path
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_sample= int(sample_rate*duration)
        self.n_mels  =n_mels
        
        #Torch audio Transformation
        
        self.resample = None
        self.mel_spectrogram = T.MelSpectrogram( sample_rate =self.sample_rate,
                                               n_fft = 2048,
                                               hop_length = 512,
                                               n_mels = self.n_mels)
        self.amplitude_to_db = T.AmplitudeToDB(top_db = 80.0)
        
    def __len__(self):
        return len(self.file_path)
    
    def _fix_audio_length_(self , waveform):
        num_samples = waveform.shape[1]
        
        if num_samples > self.target_sample:
            waveform = waveform[:,:self.target_sample]
        elif num_samples < self.target_sample:
            padding = self.target_sample - num_samples
            waveform = torch.nn.functional.pad(waveform , (0, padding))
            
        return waveform
    def __getitem__(self , idx):
        path = self.file_path[idx]
        label = self.labels[idx]
        y , _ = librosa.load(path , sr = self.sample_rate , mono=True)
        waveform = torch.tensor(y , dtype=torch.float32).unsqueeze(0)
        waveform = self._fix_audio_length_(waveform)
        mel_spec= self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        return log_mel_spec , torch.tensor(label , dtype=torch.float32)
