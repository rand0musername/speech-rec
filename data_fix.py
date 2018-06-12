import os
import glob
import numpy as np

# wav -> sph, a simple rename
def step_one():
	wav_glob = './data/TIMIT/**/*.WAV'
	for wav_filename in glob.glob(wav_glob, recursive=True):
	    sph_filename = wav_filename[:-3] + "SPH"
	    print(wav_filename)
	    os.rename(wav_filename, sph_filename)

# sph -> wav, conversion, so it can be read with scipy
def step_two():
	sph_glob = './data/TIMIT/**/*.SPH'
	for sph_filename in glob.glob(sph_glob, recursive=True):
	    print(sph_filename)
	    wav_filename = sph_filename[:-3] + "wav"
	    os.system("sox " + sph_filename + " " + wav_filename)
