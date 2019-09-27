# -*- coding: cp1252 -*-
from builtins import range
import os

import numpy as np
import librosa

import scipy

RANDOM = np.random.RandomState(1337)
CACHE = {}

def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):
    
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')

    return sig, rate

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.9)

    # Create Gaussian noise
    noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)

    return noise

def buildBandpassFilter(rate, fmin, fmax, order=4):

    global CACHE

    fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')

        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]

def applyBandpassFilter(sig, rate, fmin, fmax):

    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate, fmin, fmax)

    return scipy.signal.sosfiltfilt(filter_sos, sig)

def pcen(spec, rate, hop_length, gain=0.8, bias=10, power=0.25, t=0.060, eps=1e-6):
    s = 1 - np.exp(- float(hop_length) / (t * rate))
    M = scipy.signal.lfilter([s], [1, s - 1], spec)
    smooth = (eps + M)**(-gain)
    return (spec * smooth + bias)**power - bias**power

def get_mel_filterbanks(num_banks, fmin, fmax, f_vec, dtype=np.float32):
    '''
    An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
    an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
    flooring down the bin indices.
    '''

    global CACHE

    # Filterbank already in cache?
    fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        
        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

        # Compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # Convert mels to Hz
        banks_ends = (f_break * (10 ** (melpoints / A) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) / \
                (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) / \
                (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]

def spectrogram(sig, rate, shape=(64, 512), win_len=512, fmin=150, fmax=15000, frequency_scale='mel', magnitude_scale='nonlinear', bandpass=True, decompose=False):

    # Compute overlap
    hop_len = int(len(sig) / (shape[1] - 1)) 
    win_overlap = win_len - hop_len + 2
    #print 'WIN_LEN:', win_len, 'HOP_LEN:', hop_len, 'OVERLAP:', win_overlap

    # Adjust N_FFT?
    if frequency_scale == 'mel':
        n_fft = win_len
    else:
        n_fft = shape[1] * 2

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig, rate, fmin, fmax)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(win_len),
                                          nperseg=win_len,
                                          noverlap=win_overlap,
                                          nfft=n_fft,
                                          detrend=False,
                                          mode='magnitude')

    # Scale frequency?
    if frequency_scale == 'mel':

        # Determine the indices of where to clip the spec
        valid_f_idx_start = f.searchsorted(fmin, side='left')
        valid_f_idx_end = f.searchsorted(fmax, side='right') - 1

        # Get mel filter banks
        mel_filterbank, mel_f = get_mel_filterbanks(shape[0], fmin, fmax, f, dtype=spec.dtype)

        # Clip to non-zero range so that unnecessary multiplications can be avoided
        mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

        # Clip the spec representation and apply the mel filterbank.
        # Due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
        spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
        spec = np.dot(spec, mel_filterbank)
        spec = np.transpose(spec, [1, 0])        

    # Magnitude transformation
    if magnitude_scale == 'pcen':
        
        # Convert scale using per-channel energy normalization as proposed by Wang et al., 2017
        # We adjust the parameters for bird voice recognition based on Lostanlen, 2019
        spec = pcen(spec, rate, hop_len)
        
    elif magnitude_scale == 'log':
        
        # Convert power spec to dB scale (compute dB relative to peak power)
        spec = spec ** 2
        spec = 10.0 * np.log10(np.maximum(1e-10, spec) / np.max(spec))
        spec = np.maximum(spec, spec.max() - 100) # top_db = 100

    elif magnitude_scale == 'nonlinear':

        # Convert magnitudes using nonlinearity as proposed by Schlüter, 2018
        a = -1.2 # Higher values yield better noise suppression
        s = 1.0 / (1.0 + np.exp(-a))
        spec = spec ** s

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()
    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)

    return spec

def get_spec(sig, rate, spec_type='melspec', **kwargs):

    if spec_type.lower()== 'melspec':
        return spectrogram(sig, rate, frequency_scale='mel', **kwargs)
    else:
        return spectrogram(sig, rate, frequency_scale='linear', **kwargs)

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))
        
        sig_splits.append(split)

    return sig_splits

def specsFromSignal(sig, rate, seconds, overlap, minlen, **kwargs):

    # Split signal in consecutive chunks with overlap
    sig_splits = splitSignal(sig, rate, seconds, overlap, minlen)

    # Extract specs for every sig split
    for sig in sig_splits:

        # Get spec for signal chunk
        spec = get_spec(sig, rate, **kwargs)

        yield spec

def specsFromFile(path, rate, offset=0.0, duration=None, **kwargs):

    # Open file
    sig, rate = openAudioFile(path, rate, offset, duration)

    # Yield all specs for file
    for spec in specsFromSignal(sig, rate, **kwargs):
        yield spec