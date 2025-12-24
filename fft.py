import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fs = 44100  # sampling rate
duration = 1  # seconds per capture

while True:
    # Record from mic
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()
    signal = audio.flatten()

    # Plot waveform
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title("Waveform")

    # Compute FFT
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Take only positive half
    idx = freqs > 0
    plt.subplot(2, 1, 2)
    plt.plot(freqs[idx], np.abs(fft_vals[idx]))
    plt.title("FFT Spectrum")

    plt.pause(0.01)
    if np.mean(np.abs(signal)) > 0.6:
        print("Noise level high")
    dominant_freq = freqs[idx][np.argmax(np.abs(fft_vals[idx]))]
    print("Dominant Frequency:", dominant_freq, "Hz")


