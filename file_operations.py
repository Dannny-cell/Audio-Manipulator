# file_operations.py

import soundfile as sf
from pydub import AudioSegment

def load_audio(file_path):
    """Loads an audio file and returns the data and samplerate."""
    try:
        data, samplerate = sf.read(file_path, dtype='float32')
        return data, samplerate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def save_audio(file_path, data, samplerate):
    """Saves audio data to a file."""
    try:
        sf.write(file_path, data, samplerate)
        print(f"Audio saved to {file_path}")
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")

def convert_audio_format(input_file, output_file, output_format):
    """Converts an audio file from one format to another."""
    try:
        sound = AudioSegment.from_file(input_file)
        sound.export(output_file, format=output_format)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error during format conversion: {e}")