import pickle
import numpy as np
import multiprocessing
from utils import transform


def process_track(track, frame_length):
    samples = []
    targets = []

    # Get the mixture audio and the target stem
    mixture_audio = track.stems[0].T
    target = np.stack([track.stems[i].T for i in range(1, 5)])

    # Iterate over the audio in chunks of 'frame_length'
    for start_idx in range(0, mixture_audio.shape[1] - frame_length + 1, frame_length):
        # Extract the frames for mixture and target
        mixture_frame = transform(mixture_audio[:, start_idx:start_idx+frame_length])
        target_frame = transform(target[:,:,start_idx:start_idx+frame_length])

        samples.append(mixture_frame)
        targets.append(target_frame)

    return samples, targets

def process_audio(mus, tag, frame_length=30000):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # Map the process_track function to each track in the dataset
    results = pool.starmap(process_track, [(track, frame_length) for track in mus])

    # Flatten the list of results
    samples, targets = zip(*results)
    samples = [item for sublist in samples for item in sublist]
    targets = [item for sublist in targets for item in sublist]

    # Save the results to a file
    pickle.dump((samples, targets), open(f'{tag}.pkl', 'wb'))

    return samples, targets