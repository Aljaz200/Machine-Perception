import numpy as np


def read_data(filename):
	# reads a numpy array from a text file
	with open(filename) as f:
		s = f.read()

	return np.fromstring(s, sep=' ')


def gauss_noise(I, magnitude=.1):
	# input: grayscale image in range [0, 1], percent of corrupted pixels
	# output: modified image

	assert np.max(I) <= 1, "Image needs to be in range [0, 1]"
	assert I.dtype == np.float64, "Image needs to be in np.float64"

	return I + np.random.normal(size=I.shape) * magnitude


def sp_noise(I, percent=.1):
	# input: grayscale image in range [0, 1], percent of corrupted pixels
	# output: modified image

	assert np.max(I) <= 1, "Image needs to be in range [0, 1]"
	assert I.dtype == np.float64, "Image needs to be in np.float64"

	res = I.copy()

	res[np.random.rand(I.shape[0], I.shape[1]) < percent / 2] = 1
	res[np.random.rand(I.shape[0], I.shape[1]) < percent / 2] = 0

	return res
