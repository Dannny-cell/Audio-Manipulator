import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

def plot_waveform(data, samplerate):
    """Plots the waveform of audio data and returns the figure."""
    plot_data = data
    if data.ndim > 1:
        plot_data = data[:, 0]  # Plot left channel for stereo

    time = np.linspace(0., len(plot_data) / samplerate, len(plot_data))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, plot_data)
    ax.set_title('Waveform (Time-Domain)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    return fig

def plot_spectrum(data, samplerate):
    """Plots the frequency spectrum (FFT) and returns the figure."""
    plot_data = data
    if data.ndim > 1:
        plot_data = data[:, 0] # Plot left channel for stereo

    N = len(plot_data)
    if N == 0:
        return plt.figure() 

    yf = rfft(plot_data)
    xf = rfftfreq(N, 1 / samplerate)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(xf, np.abs(yf))
    ax.set_title('Frequency Spectrum (FFT)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    ax.set_xlim(0, samplerate / 2) 
    return fig

def plot_spectrogram(data, samplerate):
    """Plots a spectrogram and returns the figure."""
    plot_data = data
    if data.ndim > 1:
        plot_data = np.mean(data, axis=1) # Convert to mono for spectrogram

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.specgram(plot_data, Fs=samplerate, NFFT=1024, cmap='viridis')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    return fig