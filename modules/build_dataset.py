import torch
import pickle
import numpy as np
from utils import transform


def process_audio(mus, tag, frame_length=30000, max_length=10000000):
    """
    Processes an audio dataset to extract frames of audio and corresponding targets.

    :param mus: A musdb dataset object to process.
    :param frame_length: Length of each audio frame to process.
    :param max_audio_length: Maximum allowed length of audio to process.
    :return: full audio mixture and target source 
    """
    samples = []
    targets = []

    # Iterate through each track in the dataset
    for track in mus:
        # Get the mixture audio and the target stem
        mixture_audio = track.stems[0].T

        t1 = track.stems[1].T
        t2 = track.stems[2].T
        t3 = track.stems[3].T
        t4 = track.stems[4].T
        target = np.stack([t1,t2,t3,t4])

        if mixture_audio.shape[0] > max_length:
            continue

        # Iterate over the audio in chunks of 'frame_length'
        for start_idx in range(0, mixture_audio.shape[1] - frame_length + 1, frame_length):
            # Extract the frames for mixture and target

            mixture_frame = transform(mixture_audio[start_idx:start_idx+frame_length])
            target_frame = transform(target[:,:,start_idx:start_idx+frame_length])

            # Append the frames as a tuple to the dataset list
            samples.append(mixture_frame)
            targets.append(target_frame)
    
    pickle.dump((samples, targets), open(f'{tag}.pkl', 'wb'))

    return samples, targets