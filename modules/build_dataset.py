import pickle
import numpy as np
from tqdm import tqdm
from utils import transform

def process_audio(mus, tag, frame_length=30000):
    """
    Processes an audio dataset to extract frames of audio and corresponding targets.

    :param mus: A musdb dataset object to process.
    :param frame_length: Length of each audio frame to process.
    :return: full audio mixture and target source 
    """
    samples = []
    targets = []

    for track in tqdm(mus):
        # Get the mixture audio and the target stem
        mixture_audio = track.stems[0].T # shape (2, L)
        target = np.stack([track.stems[i].T for i in range(1, 5)]) # shape (4, 2, L)

        # Iterate over the audio in chunks of 'frame_length'
        for start_idx in range(0, mixture_audio.shape[1] - frame_length + 1, frame_length):
            # Extract the frames for mixture and target
            mixture_frame = transform(mixture_audio[:, start_idx:start_idx+frame_length])
            target_frame = transform(target[:,:,start_idx:start_idx+frame_length])
            samples.append(mixture_frame)
            targets.append(target_frame)

        # Iterate over the audio in chunks of 'frame_length', starting at timestep 15000
        for start_idx in range(15000, mixture_audio.shape[1] - frame_length + 1, frame_length):
            # Extract the frames for mixture and target
            mixture_frame = transform(mixture_audio[:, start_idx:start_idx+frame_length])
            target_frame = transform(target[:,:,start_idx:start_idx+frame_length])
            samples.append(mixture_frame)
            targets.append(target_frame)
    
    pickle.dump((samples, targets), open(f'{tag}.pkl', 'wb'))

    return samples, targets