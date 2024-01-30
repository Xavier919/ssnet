import torch

#add padding on both sides of samples
def pad_seqs(seqs, num_chan, num_pad=100):
    pad = torch.zeros(num_chan, num_pad)
    pad_seqs = [torch.cat([pad, x, pad], dim=1) for x in seqs]
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    X, y = zip(*Xy)
    X = pad_seqs(X, 2, num_pad=100)
    y = torch.stack(y, dim=0)
    return (X, y)

#def transform(x):
#    return torch.tensor(x).float()
import librosa
import numpy as np

def transform(x, sr=22050, n_mels=128):
    """
    Transforms an audio signal into a normalized Mel-spectrogram.

    :param x: Input audio signal (numpy array).
    :param sr: Sample rate of the audio signal.
    :param n_mels: Number of Mel bands to generate.
    :return: Transformed audio as a Mel-spectrogram.
    """
    # Normalize the audio to have zero mean and unit variance
    x_normalized = librosa.util.normalize(x)

    # Convert to Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(x_normalized, sr=sr, n_mels=n_mels)

    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Convert to PyTorch tensor
    mel_tensor = torch.tensor(mel_spec_db).float()

    return mel_tensor