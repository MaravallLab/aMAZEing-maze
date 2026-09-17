# here we will handle all things (functions) related to the sound data generation

import os
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
from config import ExperimentConfig
from scipy.signal import resample_poly
from scipy.interpolate import resample_poly
from scipy.interpolate import interp1d
from typing import List, Dict, Optional, Union, Tuple

class Audio:
    def __init__(self, cfg: ExperimentConfig, 
                 calibration_gain_path: Optional[str] = r"C:/Users/labuser/Documents/GitHub/mice-maze/code/auditory/speaker calibration scripts/frequency_response_speaker.csv"):
        
        #initialise
        self.fs = cfg.samplerate
        self.device_id = cfg.channel_id
        self.default_duration = cfg.default_sound_duration
        self.default_volume = cfg.default_volume
        self.default_ramp = cfg.default_ramp_length_s
        self.default_waveform = cfg.default_waveform
        
        #Setup Sound Card 
        sd.default.samplerate = self.fs
        sd.default.device = self.device_id

        #interpolation function from the frequency response data extrapolated from the graph on the speaker's website
        self.gain_curve = None
        if calibration_gain_path and os.path.exists(calibration_gain_path):
            try:
                data = pd.read_csv(calibration_gain_path)
                frequencies = data['Frequency_kHz'].values * 1000  # Convert kHz to Hz
                attenuation = data['Attenuation_dB'].values

                #create interpolation function: freq -> attenuation
                self.gain_curve = interp1d(frequencies, attenuation, kind='cubic', fill_value="extrapolate")
            except Exception as e:
                print(f"failed to load the frequency response function csv: {e}.\n Ensure {calibration_gain_path} exists")
        else:
            print("No gain. Just as a piece of advice, check the frequency response function for your speakers on the information sheet.")

        #little helper function to check the parameters and if the user doesn't change them, use the defaults set in config.py
    def _resolve_params(self,  waveform:Optional[str], duration: Optional[float], volume:Optional[float], ramp:Optional[float]):
        final_duration = self.default_duration if duration is None else duration
        final_ramp = self.default_ramp if ramp is None else ramp
        final_volume = self.default_volume if volume is None else volume
        final_waveform = self.default_waveform if waveform is None else waveform

        return final_waveform, final_duration, final_volume, final_ramp

    # the speaker has a non linear frequency , so we compute the gain -- increase the volume by finding the attenuation in db and 10** (-attenuation_db / 20)
    def compute_gain(self, freq_Hz: float) -> float:
        if self.gain_curve is None:
            return 1.0
        
        attenuation_db = self.gain_curve(freq_Hz)

        gain = 10** (-attenuation_db / 20)
        return gain
    
    def generate_sound_data(self, frequency: float, waveform= None, duration_s= None, volume= None, ramp_duration_s= None) -> np.ndarray:
        #function to generate sound data for a given frequency and waveform, with speaker compensation and ramping

        #Apply defaults in case the user ends up specifying these variables in main
        waveform, duration_s, volume, ramp_duration_s = self._resolve_params(waveform, duration_s, volume, ramp_duration_s)

        #if the frequency is 0, just create an array of zeros sample_rate * suration in seconds
        if frequency == 0:
            return np.zeros(int(self.fs * duration_s))
        
        # generate an array of evenly spaced numbers over an interval. In this case, array starts at 0 and ends at at duration_s, at an interval of sample rate * duration. Endpoint = False means that the stop value is not included in the sequence (because the intervals are more than the seconds obv)
        t = np.linspace(0, duration_s, int(self.fs * duration_s), endpoint = False)
        gain = self.compute_gain(frequency) 
        adjusted_volume = volume * gain

        if waveform == 'sine':
            sound = np.sin(frequency * t * 2 * np.pi)  * adjusted_volume

        elif waveform == 'square':
            sound = np.sign(np.sin(frequency * t * 2 * np.pi)) * adjusted_volume

        elif waveform == 'sawtooth':
            sound = (2 * (t * frequency % 1) - 1) * adjusted_volume

        elif waveform == 'triangle':
            sound = (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1) * adjusted_volume

        elif waveform == 'pulse wave':
            sound = np.where((t % (1 / frequency)) < (1 / frequency) * 0.5, 1.0, -1.0)  * adjusted_volume  # 0.5 in this case is the duty cycle
        
        elif waveform == 'white noise':
            samples = int(self.fs * duration_s)
            sound = np.random.uniform(low=-1.0, high=1.0, size=samples) * adjusted_volume

        return self._apply_ramp(sound, ramp_duration_s)
    
    #function to generate constant sinusoidal temporally envelope modulated data 
    def generate_simple_tem_sound_data(self, carrier_frequency:float, waveform= None, duration_s= None, volume= None, ramp_duration_s= None, modulated_frequency: float = 50.0, depth:float = 0.5) -> np.ndarray: 
        
        #Apply defaults in case the user ends up specifying these variables in main
        waveform, duration_s, volume, ramp_duration_s = self._resolve_params(waveform, duration_s, volume, ramp_duration_s)

        #we generate the sound but we add the ramp at the end in case
        sound = self.generate_sound_data(carrier_frequency, waveform=waveform, duration_s = duration_s, volume = volume, ramp_duration_s= 0.0)

        #generate envelope
        t = np.linspace(0, duration_s, len(sound), endpoint = False)
        envelope = 1 + depth * np.sin(2 * np.pi * modulated_frequency * t)
        modulated_sound = sound * envelope

        #we reapply the ramp at the end
        return self._apply_ramp(modulated_sound, ramp_duration_s)
    

    # function to generate complex temporal envelope modulated sounds
    def generate_complex_tem_sound_data(self, carrier_frequency: float, waveform = None, duration_s= None, volume = None , ramp_duration_s= None, modulated_frequencies_list: List[float]= [30, 50, 70], depth:float = 0.5) -> np.ndarray:
        #Apply defaults in case the user ends up specifying these variables in main
        waveform, duration_s, volume, ramp_duration_s = self._resolve_params(waveform, duration_s, volume, ramp_duration_s)

        #we generate the sound but we add the ramp at the end in case
        sound = self.generate_sound_data(carrier_frequency, waveform=waveform, duration_s = duration_s, volume = volume, ramp_duration_s= 0.0)
        t = np.linspace(0, duration_s, len(sound), endpoint = False)

        #apply different modulation in different segments
        # Apply different modulations in segments
        segment_len = int(0.2 * self.fs) # 200ms segments
        num_segments = int(np.ceil(len(sound) / segment_len))
        
        # Cycle through the modulation frequencies
        mod_sequence = (modulated_frequencies_list * ((num_segments // len(modulated_frequencies_list)) + 1))[:num_segments]
        
        modulated = np.copy(sound)
        
        for i, f_mod in enumerate(mod_sequence):
            start = i * segment_len
            end = min((i + 1) * segment_len, len(sound))
            
            # Local time vector for this segment to keep phase continuous-ish
            t_seg = t[start:end]
            envelope = 1 + 0.5 * np.sin(2 * np.pi * f_mod * t_seg)
            modulated[start:end] *= envelope
            
        # Normalize to prevent clipping
        max_val = np.max(np.abs(modulated))
        if max_val > 1.0:
            modulated /= max_val
            
        return self._apply_ramp(modulated, ramp_duration_s)
    

    def load_wav(self, path: str) -> np.ndarray:
        # load a wav and resample to the system samplerate
        if not os.path.exists(path):
            print(f"path {path} not found, no sound played")
            return np.zeros(int(self.fs * 1.0))

        data, fs_original = sf.read(path)
        if data.ndim > 1: data = np.mean(data, axis=1)
        if fs_original != self.fs:
            from math import gcd
            g = gcd(self.fs, fs_original)
            data = resample_poly(data, self.fs // g, fs_original // g)
        return data

    def mix_sounds(self, sound1: np.ndarray, sound2: np.ndarray) -> np.ndarray:
        min_len = min(len(sound1), len(sound2))
        mixed = sound1[:min_len] + sound2[:min_len]
        #normalise
        max_val = np.max(np.abs(mixed))
        if max_val > 0: mixed /= max_val
        return mixed


    def play(self, sound_data: np.ndarray):
        #stop previous sound
        sd.stop()
        # Safety Check
        sound_data = np.asarray(sound_data)
        if sound_data.ndim > 1:
            sound_data = np.mean(sound_data, axis=1)
            
        sd.play(sound_data, self.fs)

    def stop(self):
        sd.stop()

    def _apply_ramp(self, sound: np.ndarray, ramp_sec: float) -> np.ndarray:
        #Applies a fade-in (and optionally fade-out) to avoid clicking.
        ramp_samples = int(self.fs * ramp_sec)
        if ramp_samples <= 0 or ramp_samples > len(sound):
            return sound
            
        envelope = np.ones_like(sound)
        envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
        # envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples) # Optional fade out
        
        return sound * envelope



