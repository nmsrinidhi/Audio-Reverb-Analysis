import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import soundfile as sf
import os

# Load your audio
y, sr = librosa.load("audio.wav")

# Create a fake impulse response (simulated room echo)
taps = np.zeros(1000)
taps[0] = 1       # direct sound
taps[150] = 0.6   # echo 1
taps[400] = 0.3   # echo 2
taps[700] = 0.1   # echo 3

# Plot impulse response
plt.figure(figsize=(6, 2))
plt.plot(taps)
plt.title("Simulated Impulse Response")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
print("hi")

# Apply the impulse response (add reverb)
reverb_audio = fftconvolve(y, taps, mode='full')
print(reverb_audio)

# Normalize audio (avoid clipping)
reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))

# Save the new audio file
output_path = "audio_with_reverb.wav"
sf.write(output_path, reverb_audio, sr)

# Confirm file saved
if os.path.exists(output_path):
    print(f"File saved as: {output_path}")
else:
    print("Something went wrong! File was not saved.")

# Plot original vs reverb waveform
plt.figure(figsize=(12, 3))
plt.plot(y[:15000], label="Original")
plt.plot(reverb_audio[:15000], label="With Reverb", alpha=0.7)
plt.legend()
plt.title("Waveform: Original vs Reverb Audio")
plt.tight_layout()
plt.show()

# Spectrogram comparison
plt.figure(figsize=(12, 5))

# Original
plt.subplot(1, 2, 1)
D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D1, sr=sr, x_axis='time', y_axis='hz')
plt.title("Original Audio")

# Reverb
plt.subplot(1, 2, 2)
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(reverb_audio)), ref=np.max)
librosa.display.specshow(D2, sr=sr, x_axis='time', y_axis='hz')
plt.title("Reverb Audio")

plt.tight_layout()
plt.show()

print("end")
