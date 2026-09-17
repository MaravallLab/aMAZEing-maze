import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import time


#This is 
sd.default.samplerate = 192000 #available sample rates: 44.1kHz, 48kHz , 88.2kHz, 96kHz, 176.4kHz , 192 kHz
sd.default.device = 3

def generate_sound_data(frequency, volume=1, duration=10, fs=192000):
    """Generate a sine wave sound with the given frequency, volume, and duration."""
    t = np.linspace(0, duration, int(fs * duration), False)
    sound = np.sin(2 * np.pi * frequency * t) * volume
    return sound

def play_sound(sound_data, fs=192000, volume=1.0):
    """Play sound using sounddevice library with improved volume scaling."""
    # sound_data = np.clip(sound_data, -1, 1)  # Ensure within valid range
    # sound_data_scaled = np.int16(sound_data * 32767 * volume)  # Apply volume scaling in dB range
    # sd.play(sound_data_scaled, fs)
    # sd.wait()
    sd.play(sound_data)
    sd.wait()


def start_calibration():
    try:
        sample_rate = int(sample_rate_entry.get())
        num_sounds = int(num_sounds_entry.get())
        sounds = []

        for i in range(num_sounds):
            freq = float(freq_entries[i].get())*1000 #convert Hz to KHz
            vol = float(vol_entries[i].get())
            sound = generate_sound_data(freq, volume=vol, fs=sample_rate)
            sounds.append((sound,vol))

        messagebox.showinfo("Info", "Playing sounds for calibration...")
        
        for i, (sound,vol) in enumerate(sounds):
            play_sound(sound, sample_rate, volume=vol)
            time.sleep(1)  # Pause between sounds
        
        messagebox.showinfo("Info", "Calibration Complete!")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

def update_sound_inputs():
    for widget in sound_frame.winfo_children():
        widget.destroy()
    
    global freq_entries, vol_entries
    freq_entries = []
    vol_entries = []
    num_sounds = int(num_sounds_entry.get())

    for i in range(num_sounds):
        tk.Label(sound_frame, text=f"Sound {i+1} Frequency (KHz):").grid(row=i, column=0)
        freq_entry = tk.Entry(sound_frame)
        freq_entry.grid(row=i, column=1)
        freq_entries.append(freq_entry)

        tk.Label(sound_frame, text=f"Sound {i+1} Volume (0-1):").grid(row=i, column=2)
        vol_entry = tk.Entry(sound_frame)
        vol_entry.grid(row=i, column=3)
        vol_entry.insert(0,"1")
        vol_entries.append(vol_entry)

# Create main UI window
root = tk.Tk()
root.title("Speaker Calibration Tool")
root.geometry("500x400")

# Sample rate input
tk.Label(root, text="Sample Rate (Hz):").pack()
sample_rate_entry = tk.Entry(root)
sample_rate_entry.pack()
sample_rate_entry.insert(0, "192000")  # Default value

# Number of sounds input
tk.Label(root, text="Number of Sounds:").pack()
num_sounds_entry = tk.Entry(root)
num_sounds_entry.pack()
num_sounds_entry.insert(0, "3")  # Default value

# Button to update sound inputs
tk.Button(root, text="Set Number of Sounds", command=update_sound_inputs).pack()

# Frame for frequency and volume inputs
sound_frame = tk.Frame(root)
sound_frame.pack()
update_sound_inputs()

# Start Calibration Button
tk.Button(root, text="Start Calibration", command=start_calibration).pack()

# Run the UI
root.mainloop()
