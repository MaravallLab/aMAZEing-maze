This directory includes the scripts you need to calibrate your speaker, if it also has a non-flat frequency response. 

We tried 2 approaches: 
1. The approach that we ended up using is in the second half of notebook_calibration.ipynb. We retrieved the datapoints for the speaker frequency response using an online tool called WebPlotDigitizer (https://automeris.io/), and computed the gain to flatten the frequency response. Obviously make sure to replace the path to the .csv file with the path to your .csv file. Unless you are also using the Avisoft Vifa Ultrasonic speaker. 
2. The approach we tried and tested so you don't have to is: generate sounds by running calibration_freq.py and record them with an ultrasonic microphone and a digital sound meter, found a calibration constant to map FFT amplitude of the frequency to a volume in dB SPL. Ensure to change the sample rate of the output to the sample rate used by the sound card. Make sure to have a look at fft_volume and notebook_calibration.ipynb to modify according to your data/sound recording.

