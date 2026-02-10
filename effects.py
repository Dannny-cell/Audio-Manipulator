import numpy as np
import librosa
from scipy.signal import butter, lfilter

def apply_panning(data, samplerate, pan_value):
    """
    Applies panning to stereo audio data.
    pan_value: -1 (full left) to 1 (full right).
    """
    if data.ndim != 2:
        # Panning requires stereo, so do nothing if mono
        return data

    # Constant power panning
    pan_rad = np.deg2rad((pan_value + 1) * 45)
    left_gain = np.cos(pan_rad)
    right_gain = np.sin(pan_rad)

    panned_data = np.copy(data)
    panned_data[:, 0] *= left_gain  # Left channel
    panned_data[:, 1] *= right_gain # Right channel
    
    return panned_data

def apply_pitch_shift(data, samplerate, n_steps):
    """
    Applies pitch shifting.
    n_steps: Number of semitones to shift.
    """
    if data.ndim == 2:
        # Process channels separately
        left_shifted = librosa.effects.pitch_shift(y=data[:, 0], sr=samplerate, n_steps=n_steps)
        right_shifted = librosa.effects.pitch_shift(y=data[:, 1], sr=samplerate, n_steps=n_steps)
        return np.stack([left_shifted, right_shifted], axis=1)
    else:
        return librosa.effects.pitch_shift(y=data, sr=samplerate, n_steps=n_steps)

def apply_time_stretch(data, samplerate, rate):
    """
    Applies time stretching without changing pitch.
    rate: Stretch factor. >1 speeds up, <1 slows down.
    """
    if data.ndim == 2:
        # Process channels separately
        left_stretched = librosa.effects.time_stretch(y=data[:, 0], rate=rate)
        right_stretched = librosa.effects.time_stretch(y=data[:, 1], rate=rate)
        return np.stack([left_stretched, right_stretched], axis=1)
    else:
        return librosa.effects.time_stretch(y=data, rate=rate)

def apply_delay(data, samplerate, delay_ms=200, feedback=0.5, mix=0.5):
    """
    Applies a simple feedback delay effect.
    """
    delay_samples = int(samplerate * (delay_ms / 1000.0))
    
    output = np.copy(data)
    
    if data.ndim == 1:
        # Mono
        for i in range(delay_samples, len(data)):
            output[i] += output[i - delay_samples] * feedback
    else:
        # Stereo
        for i in range(delay_samples, len(data)):
            output[i, 0] += output[i - delay_samples, 0] * feedback
            output[i, 1] += output[i - delay_samples, 1] * feedback
            
    # Mix wet and dry signals
    mixed_data = (1 - mix) * data + mix * output

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed_data))
    if max_val > 0:
        mixed_data /= max_val

    return mixed_data.astype(np.float32)
