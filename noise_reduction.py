import librosa
import numpy as np

def _reduce_noise_mono(data, samplerate, reduction_factor=1.5):
    """Reduces noise from a single mono channel."""
    # Perform Short-Time Fourier Transform (STFT)
    stft_data = librosa.stft(data)
    magnitude, phase = librosa.magphase(stft_data)

    # Estimate noise profile from a small initial chunk of the audio
    # Ensure we don't try to sample more frames than exist in short clips
    noise_profile_frames = min(int(samplerate / 10), magnitude.shape[1])
    noise_profile = np.mean(magnitude[:, :noise_profile_frames], axis=1)

    # Create a mask to filter out noise
    # The mask is True for frequencies above the noise profile threshold
    mask = (magnitude.T > noise_profile * reduction_factor).T
    
    # Apply the mask to the magnitude
    denoised_magnitude = magnitude * mask
    
    # Inverse STFT to get back to time-domain signal
    # Ensure the output length matches the input length for consistency
    return librosa.istft(denoised_magnitude * phase, length=len(data))

def reduce_noise_spectral_gating(data, samplerate, reduction_factor=1.5):
    """
    Reduces noise using spectral gating, handling both mono and stereo audio.
    """
    if data.ndim == 1:
        # It's a mono file, process it directly
        return _reduce_noise_mono(data, samplerate, reduction_factor).astype(np.float32)
    elif data.ndim == 2:
        # It's a stereo file, process each channel separately
        left_channel_denoised = _reduce_noise_mono(data[:, 0], samplerate, reduction_factor)
        right_channel_denoised = _reduce_noise_mono(data[:, 1], samplerate, reduction_factor)
        # Stack the processed channels back together into a stereo array
        return np.stack([left_channel_denoised, right_channel_denoised], axis=1).astype(np.float32)
    
    # If format is unexpected (e.g., more than 2 channels), return original data
    return data