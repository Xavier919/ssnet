import musdb
import os
import argparse
from modules.build_dataset import process_audio

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('frame_length', type=int)
parser.add_argument('max_length', type=int)
args = parser.parse_args()


if __name__ == "__main__":

    # Set the MUSDB_PATH environment variable
    os.environ['MUSDB_PATH'] = args.data_path
    # Initialize the musdb datasets
    mus_train = musdb.DB(subsets="train", split='train')
    mus_valid = musdb.DB(subsets="train", split='valid')
    mus_test = musdb.DB(subsets="test")

    FRAME_LENGTH = args.frame_length
    MAX_LENGTH = args.max_length

    X_train, y_train = process_audio(mus_train, tag='train', frame_length=FRAME_LENGTH, max_length=MAX_LENGTH)
    X_valid, y_valid = process_audio(mus_valid, tag='valid', frame_length=FRAME_LENGTH, max_length=MAX_LENGTH)
    X_test, y_test = process_audio(mus_test, tag='test', frame_length=FRAME_LENGTH, max_length=MAX_LENGTH)