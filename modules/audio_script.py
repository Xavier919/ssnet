from scipy.io.wavfile import write
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    test = pickle.load(open(args.test, 'rb'))
    X_test, y_test = test

    

    #sample_rate = 44100  
    #wav_file = "output.wav"
    #write(wav_file, sample_rate, audio_data) 