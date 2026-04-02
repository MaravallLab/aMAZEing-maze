# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile  # For WAV file handling
from scipy.fftpack import fft  # Fast Fourier Transform
from scipy.signal import get_window  # Window functions
import tkinter as tk
from tkinter import filedialog as fd

# File selection dialog using Tkinter
def choose_csv():
    root = tk.Tk()
    filename = fd.askopenfilename()  # Open file dialog
    root.destroy()  # Close the Tkinter window
    return filename

def spectrogram_n_fft(INPUT_FILE, SPEED_RATE, show_plot = True):
    # --- Audio File Handling ---
    sample_rate, data = wavfile.read(INPUT_FILE)
    
    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data[:, 0]  # Take first channel

    # --- Analysis Parameters ---
    window_size = 8192  # Samples per FFT (affects frequency resolution)
    hop_size = 512      # Samples between FFTs (affects time resolution)
    window = get_window('hann', window_size)  # Windowing function
    duration = len(data) / sample_rate  # Total audio duration

    # --- Plot Setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))
    
    # --- Spectrogram Plot ---
    Pxx, freqs, bins, im = ax1.specgram(data,
                                        NFFT=window_size, 
                                        Fs=sample_rate,
                                        noverlap=window_size - hop_size,
                                        cmap='viridis')
    ax1.set_title('Spectrogram')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')
    
    # Configure grid and ticks
    ax1.set_xticks(np.arange(0, duration, 5))  # Major ticks every 5s
    ax1.set_xticks(np.arange(0, duration, 1), minor=True)  # Minor every 1s
    ax1.grid(which='major', alpha=0.5)
    ax1.grid(which='minor', alpha=0.2)

    # --- Sound Event Detection ---
    try:
        # Calculate RMS energy for each window
        rms = []
        for i in range(0, len(data)-window_size, hop_size):
            segment = data[i:i+window_size].astype(np.float64)
            rms.append(np.sqrt(np.mean(segment**2)))
        
        rms = np.array(rms)
        times = np.arange(len(rms)) * hop_size / sample_rate
        
        # Dynamic threshold calculation (95th percentile based)
        threshold = 0.1 * np.percentile(rms, 95)
        events = rms > threshold
        
        # Event boundary detection
        event_changes = np.diff(events.astype(int))
        event_starts = np.where(event_changes == 1)[0] + 1
        event_ends = np.where(event_changes == -1)[0] + 1

        # Edge case handling
        if events[0]:
            event_starts = np.insert(event_starts, 0, 0)
        if events[-1]:
            event_ends = np.append(event_ends, len(events)-1)

        start_times = []
        end_times = []
        peak_freqs = []

        # Visualize events on spectrogram
        for start_idx, end_idx in zip(event_starts, event_ends):
            start_time = times[start_idx]
            start_times.append(round(start_time, 2))
            end_time = times[end_idx] + (window_size/sample_rate)
            end_times.append(round(end_time,2))
            

            # Extract dominant frequency (highest power)
            event_spectrum = np.mean(Pxx[:, start_idx:end_idx], axis=1)
            dominant_freq_idx = np.argmax(event_spectrum)
            peak_freqs.append(round(freqs[dominant_freq_idx], 2))


            # Draw event regions
            ax1.axvspan(start_time, end_time, color='magenta', alpha=0.3)
            
            # Add time labels
            ax1.text(start_time + 0.05, ax1.get_ylim()[1]*0.95, 
                    f'{start_time:.2f}s', color='white', 
                    fontsize=9, rotation=90, va='top')
            ax1.text(end_time - 0.15, ax1.get_ylim()[1]*0.95,
                    f'{end_time:.2f}s', color='white',
                    fontsize=9, rotation=90, va='top')
            


    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # --- Real-Time FFT Visualization ---
    line, = ax2.plot([], [], lw=2)
    ax2.set_xlim(0, 45000)  # Frequency range (up to 45kHz)
    ax2.set_ylim(0, 1)      # Normalized amplitude
    
    # Visual elements
    vline = ax2.axvline(0, color='r', linestyle='--', alpha=0.7)
    time_text = ax2.text(0.02, 0.8, '', transform=ax2.transAxes)

    # Animation functions
    def init():
        line.set_data([], [])
        vline.set_visible(False)
        return line, vline,

    def update(frame):
        start = frame * hop_size * SPEED_RATE 
        end = start + window_size
        
        if end > len(data):
            return line,
        
        # FFT processing
        segment = data[start:end] * window
        spectrum = np.abs(fft(segment))[:window_size//2]
        spectrum /= np.max(spectrum)  # Normalize
        
        freqs = np.linspace(0, sample_rate//2, window_size//2)
        peak_idx = np.argmax(spectrum)
        current_time = start / sample_rate
        
        # Update display
        line.set_data(freqs, spectrum)
        vline.set_xdata(freqs[peak_idx])
        time_text.set_text(f'Time: {current_time:.2f}s\nPeak: {freqs[peak_idx]:.1f} Hz')
        
        return line, vline, time_text 
    # Animation control
    frames = (len(data) - window_size) // (hop_size * SPEED_RATE)
    ani = animation.FuncAnimation(
        fig, update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=(hop_size/sample_rate * 1000)/SPEED_RATE
    )

    if show_plot:
        plt.tight_layout()
        plt.show()
    return start_times, end_times, peak_freqs


def get_start_end_times(INPUT_FILE, freq, spls_for_all, times):

    for i in range(1,5):
        if str(i) in INPUT_FILE:
            specific_times = times[i-1]
            start_time = specific_times[:,0]
            end_time = specific_times[:,1]

            start_time = [float(i) for i in start_time]
            end_time = [float(i) for i in end_time]

            spls = spls_for_all[:, i-1]


    return start_time, end_time, spls


def find_amplitude(input_wav, start_time, end_time, N, target_freq):
    sample_rate, data = wavfile.read(input_wav)
    if data.ndim > 1:
        data = data[:, 0]  # Use one channel if stereo

    # Define the time range where the 16 kHz tone is steady

    start_index = int(start_time * sample_rate)
    end_index = int(end_time * sample_rate)
    segment = data[start_index:end_index]

    # FFT parameters
                    # FFT length
    hop_size = N // 2          # 50% overlap
    window = get_window('hann', N)
    window_cg = np.mean(window)  # Coherent gain (≈0.5 for Hann window)
    window = window/ window_cg

    # Create the frequency axis for one block
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)

    # Find the FFT bin closest to 16 kHz
    target_idx = np.argmin(np.abs(freqs - target_freq))

    # Initialize list to collect amplitude values
    amplitudes = []
    amplitudes_raw = []

    # Slide the window across the segment
    for start in range(0, len(segment) - N + 1, hop_size):
        block = segment[start:start + N]
        # Apply the Hann window to reduce spectral leakage
        windowed_block = block * window
        # Compute the FFT of the windowed block
        fft_block = np.fft.rfft(windowed_block)
        # Extract the raw amplitude at the target frequency
        amp_raw = np.abs(fft_block[target_idx])
        # Correct for FFT scaling and window attenuation
        amp = (2.0 / N) * (amp_raw)
        amplitudes.append(amp)
        amplitudes_raw.append(amp_raw)

    # Compute the average amplitude at 16 kHz
    A16_avg = np.mean(amplitudes)
    A16_avg_raw = np.mean(amplitudes_raw)
    #print(f"Average amplitude at {target_freq}Hz: {A16_avg} / raw:{A16_avg_raw}")

    return A16_avg, A16_avg_raw

def find_calibration_constant(SPL, amplitude):
    C= SPL - 20*(np.log10(amplitude)) 
    return C

#print (find_calibration_constant(64.15, 38314.66638806413))



def find_c_for_all (INPUT_FILE, freq, start_time, end_time, spls):

    Amps = []
    calibration_constants_lists = []
    for i in range(len(freq)):
        amp, amp_raw = find_amplitude(INPUT_FILE, start_time[i], end_time[i], 8192, freq[i])
        cc= find_calibration_constant(spls[i], amp)

        Amps.append(amp)
        calibration_constants_lists.append(cc)

    return calibration_constants_lists , Amps


def find_time_range(INPUT_FILE, show_plot = True):
    
    #check the spectrogram and normalised fft to determine the time window to extract the amplitude
    SPEED_RATE = 5
    # if not SPEED_UP_FFT:
    #     SPEED_RATE = 1
    #fv.spectrogram_n_fft(INPUT_FILE, SPEED_RATE)
    start_times, end_times, freqs = spectrogram_n_fft(INPUT_FILE, SPEED_RATE, show_plot)
    clean_start = []
    clean_end= []
    clean_freq=[]

    #print(len(start_times), len(end_times), len(freqs))

    for i in range(len(start_times)): 
        time_len = end_times[i] - start_times[i]
        if time_len >= 5:
            clean_start.append(start_times[i])
            clean_end.append(end_times[i])
            clean_freq.append(freqs[i])
            
    start_times, end_times, freqs = clean_start, clean_end, clean_freq 
    freqs = sum(freqs)/len(freqs)

    return start_times, end_times, freqs
