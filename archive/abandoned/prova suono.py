import tkinter as tk
from tkinter import simpledialog, Scale, HORIZONTAL, Label, ttk, filedialog, messagebox, Entry, font
import csv
import numpy as np
import sounddevice as sd
import wave
import time
import os

#this code essentially creates sounds 

class SoundSynthesizerApp:
    def __init__(self, root, csvfile):
        #root for the window
        self.root = root
        #name of the window
        self.root.title("Sound Synthesizer for ROIs")
        #show the rois from the csv file
        self.rois = self.read_rois(csvfile)
        #number the rois
        self.num_rois = len(self.rois)

        # Frequency Control
        Label(root, text="Frequency (Hz)").pack()
        self.freq_entry = Entry(root)
        self.freq_entry.pack()

        # Volume Control
        Label(root, text="Volume").pack()
        # setting the sliding thingy for the volume
        self.volume_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
        self.volume_slider.pack()

        # Waveform Type Dropdown
        Label(root, text="Waveform").pack()
        self.waveform_var = tk.StringVar(root)
        self.waveform_var.set("sine")  # default value
        self.waveform_options = ['sine', 'square', 'sawtooth', 'triangle', 'pulse wave', 'white noise']
        self.waveform_dropdown = ttk.Combobox(root, textvariable=self.waveform_var, values=self.waveform_options)
        self.waveform_dropdown.pack()

        # ROI Dropdown
        Label(root, text="Select ROI Index").pack()
        self.selected_roi_var = tk.StringVar(root)
        self.selected_roi_var.set(1)  # default value
        self.roi_options = [i+1 for i in range(self.num_rois)]
        self.roi_dropdown = tk.OptionMenu(root, self.selected_roi_var, *self.roi_options)
        self.roi_dropdown.pack()

        # Play Sound Button
        self.play_sound_btn = tk.Button(root, text="Play Sound", command=self.play_sound)
        self.play_sound_btn.pack()



    def read_rois(self, csvfile):
        rois = []
        with open(csvfile, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                rois.append(row)
        return rois

    def play_sound(self):
        frequency = float(self.freq_entry.get())
        volume = self.volume_slider.get() / 100  # Scale volume to 0-1
        waveform = self.waveform_var.get()
        sound_data = self.generate_sound_data(frequency, volume, waveform)
        self.play_sound_data(sound_data)
        self.ask_to_save_sound(sound_data, frequency, waveform)

    def generate_sound_data(self, frequency, volume, waveform):
        duration = 10  # seconds
        fs = 44100  # Sample rate
        t = np.linspace(0, duration, int(fs * duration), False)

        if waveform == 'sine':
            sound = np.sin(frequency * t * 2 * np.pi)
        elif waveform == 'square':
            sound = np.sign(np.sin(frequency * t * 2 * np.pi))
        elif waveform == 'sawtooth':
            sound = 2 * (t * frequency % 1) - 1
        elif waveform == 'triangle':
            sound = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        elif waveform == 'pulse wave':
            t = np.linspace(0, duration, int(fs * duration), False)
            sound= np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0) #0.5 in this case is the duty cycle
        elif waveform == 'white noise':
            samples = int(fs * duration)
            sound = np.random.uniform(low=-1.0, high=1.0, size=samples)

        return sound * volume

    def play_sound_data(self, sound_data):    
        sd.play(sound_data, samplerate=44100)
        sd.wait()

    def ask_to_save_sound(self, sound_data, frequency, waveform):
        #asks user if they want to save sound
        if tk.messagebox.askyesno("Save Sound", "Do you want to save this sound?"):

            #user choses directory where the subdirectories are going to be created--- not going to ask to choose a directory a second time
            if not hasattr (self, 'parent_folder_selected'):
                self.parent_folder_selected = filedialog.askdirectory()

            if self.parent_folder_selected:
                local_time = time.localtime()
                date_time = time.strftime('%d-%m-%Y_%H', local_time) #adding hour in case we of trial/error 

                # Create a new subdirectory within the selected folder that takes the day
                subfolder_path = os.path.join(self.parent_folder_selected, date_time)

                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

                #stores the parent path for future saves within the same session
                #self.current_session_folder=self.parent_folder_selected

                #get the ROI index --this was defined in __init__ method
                roi_index = self.selected_roi_var.get()

                #use the roi index as the name for the sound
                file_name= f"{roi_index}.wav"

                #save in the folder
                full_file_path= os.path.join (subfolder_path, file_name)
                self.save_sound_to_file(sound_data, full_file_path)

        


    def save_sound_to_file(self, sound_data, file_path):
        '''Modify this if need different audio formats'''
        # Normalize sound_data to 16-bit PCM format for WAV file
        sound_data_normalized = np.int16((sound_data / sound_data.max()) * 32767)
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(44100) #sampling rate
            wf.writeframes(sound_data_normalized.tobytes())

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundSynthesizerApp(root, "auditory_rois.csv")  # Update the path to your CSV file
    root.mainloop()
