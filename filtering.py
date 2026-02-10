from scipy.signal import butter, lfilter, sosfilt
import numpy as np

def _process_channel(func, data, *args, **kwargs):
    """Helper to apply a function to both channels of stereo audio."""
    if data.ndim == 1:
        # It's already mono, just process it
        return func(data, *args, **kwargs)
    elif data.ndim == 2:
        # Process left and right channels separately and stack them back
        left = func(data[:, 0], *args, **kwargs)
        right = func(data[:, 1], *args, **kwargs)
        return np.stack([left, right], axis=1)
    return data

def _apply_eq_mono(data, samplerate, bass_gain, mid_gain, treble_gain):
    """Applies a 3-band EQ to a single mono channel."""
    bass_freq, mid_freq_low, mid_freq_high, treble_freq = 250, 251, 4000, 4001
    
    # Convert dB gain to a linear multiplier
    bass_lin = 10**(bass_gain / 20.0)
    mid_lin = 10**(mid_gain / 20.0)
    treble_lin = 10**(treble_gain / 20.0)
    
    nyquist = 0.5 * samplerate
    
    # Design filters for each band
    low_shelf = butter(5, bass_freq / nyquist, btype='low', output='sos')
    mid_pass = butter(5, [mid_freq_low / nyquist, mid_freq_high / nyquist], btype='band', output='sos')
    high_shelf = butter(5, treble_freq / nyquist, btype='high', output='sos')
    
    # Apply each filter and its gain
    bass_part = sosfilt(low_shelf, data) * bass_lin
    mid_part = sosfilt(mid_pass, data) * mid_lin
    treble_part = sosfilt(high_shelf, data) * treble_lin

    # Combine the bands
    processed_data = bass_part + mid_part + treble_part
    return processed_data

def apply_eq(data, samplerate, bass_gain, mid_gain, treble_gain):
    """Applies a 3-band EQ, handling both mono and stereo audio."""
    processed = _process_channel(_apply_eq_mono, data, samplerate, bass_gain, mid_gain, treble_gain)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed /= max_val
        
    return processed.astype(np.float32)

def _apply_filter_mono(data, samplerate, cutoff_freq, order=5, filter_type='lowpass'):
    """Applies a standard filter to a single mono channel."""
    nyquist = 0.5 * samplerate
    
    if filter_type == 'bandpass':
        low = cutoff_freq[0] / nyquist
        high = cutoff_freq[1] / nyquist
        b, a = butter(order, [low, high], btype=filter_type, analog=False)
    else:
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        
    return lfilter(b, a, data)

def apply_filter(data, samplerate, cutoff_freq, order=5, filter_type='lowpass'):
    """Applies a filter, handling both mono and stereo audio."""
    processed = _process_channel(_apply_filter_mono, data, samplerate, cutoff_freq, order, filter_type)
    return processed.astype(np.float32)

