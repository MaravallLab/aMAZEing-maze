# Troubleshooting

Problems and their fixes, gathered from every page and grouped by when they show up. Each links to the page with the full story.

## Installing and starting

??? question "The download link says Not Found"
    No release of the application has been published yet. Install [from source](../getting-started/installation.md#from-source), or ask the lab for a copy of the zip.

??? question "Windows says it protected your PC"
    The application is not signed with a paid certificate. Click **More info**, then **Run anyway**. Windows asks only the first time. [Download page](../getting-started/download.md#if-windows-says-it-protected-your-pc)

??? question "The application does not start, or says a file is missing"
    It must be extracted from the zip first, with the `_internal` folder kept beside `amazeing-app.exe`. The first start can take up to a minute. [If it fails](../getting-started/download.md#if-it-fails)

??? question "`amaze-app` is not recognised as a command"
    The package is not installed in the Python you are using, or that Python's `Scripts` folder is not on the PATH. Activate the environment you installed into, or run `python -m amazeing.app`. [Installation](../getting-started/installation.md#from-source)

??? question "How do I know the installation works, without a camera or sound card?"
    `amaze-auditory --write-config check.yaml` writes a configuration file and stops, touching no hardware. If the file appears, the installation is sound. With the standalone application: `amazeing-app.exe --entry auditory --write-config check.yaml`.

## Setting up a session

??? question "The camera shows the wrong room"
    The camera index is wrong. With a built-in webcam as well, the maze camera is usually `1`. Change **Camera index** in **Devices** and press **Check camera** again. [First session, step 2](../getting-started/first-session.md#step-2-tell-it-how-many-arms-and-where-they-are)

??? question "Nothing in the audio device list looks right"
    Run `python -c "import sounddevice; print(sounddevice.query_devices())"`. The numbers on the left are the indices. If the interface is missing altogether, it is a driver problem. [First session, step 1](../getting-started/first-session.md#step-1-point-it-at-your-hardware)

??? question "I press Play and hear nothing"
    Ultrasonic stimuli are inaudible, which is expected. To check the device, set one arm to an audible frequency such as 8000 Hz for a moment. [First session, step 5](../getting-started/first-session.md#step-5-look-at-the-sounds-before-playing-them)

??? question "An arm bar is amber or red with the maze empty"
    Amber: the arm sits close to the line and will trip on a shadow; lower the sensitivity or fix the lighting. Red with nothing there: the baseline was measured with something in the maze; press ++c++ to measure it again with the maze empty. [First session, step 3](../getting-started/first-session.md#step-3-check-the-detection)

??? question "Boxes turn red with nothing in the arm, or stay blue when something is there"
    Adjust **Detection sensitivity**: raise it to detect more readily, lower it to be stricter. Check that the lighting is even and the room dark: stray visible light makes the brightness drift. [Build the rig](../hardware/build.md)

??? question "The form says the session will not start: stimuli and arms disagree"
    Some modes build a fixed number of stimuli, which must match the number of arms. The **Stimuli** line under Experiment gives the arithmetic. [Experiment modes](../guide/experiment-modes.md#the-stimuli-must-fill-the-arms)

??? question "The form says a sound file is missing"
    A mode that plays a recording cannot find it. Set the file in **Vocalisation files**, or untick the vocalisation arm in the mode's control arms. A missing file would otherwise give a silent arm where a recording was meant. [Experiment modes](../guide/experiment-modes.md#recordings-the-mode-needs)

??? question "The tactile servos do not move, or move to the wrong angle"
    Check the gratings' angles against your rig before anything else; the shipped `grating_maps.csv` is a template. `firmware/arduino/i2c_scanner/` confirms the servo board is seen. To run the task without the Arduino, set `serialOn = False` at the top of `simplerCode.py`. [Tactile paradigm](../guide/tactile.md) and [Firmware](../hardware/firmware.md)

## After a session

??? question "Every arm shows zero time"
    Detection failed. The baseline was probably measured with something in the maze, or the arm boxes (`rois1.csv`) no longer match the camera position. Check `session_manifest.json` for the settings used, and compare the boxes with a frame of the video. [Analysis tutorial](../analysis/tutorial.md#step-1-check-one-session)

??? question "A handful of visits, or thousands"
    A handful in an hour means detection was too strict; thousands means it was too loose and flickering. Redo the detection check. [Analysis tutorial](../analysis/tutorial.md#step-1-check-one-session)

??? question "The summary tools cannot group sessions by day"
    The sessions were recorded without a day label, so they sit directly under the mode folder. Set **Day label** in the application, or `--day` on the command line, for future sessions. [Analysis tutorial](../analysis/tutorial.md#what-you-should-have)

??? question "Visit times look a thousand times too large or too small"
    The first version of the software recorded `time_spent` in milliseconds; the current one records seconds. `session_manifest.json` states the unit of every column. [Data dictionary](data-dictionary.md#units-and-versions)

??? question "Visits in old data are implausibly long"
    The first version did not close a visit that ran over the end of a block. The loader of the analysis scripts works around it. [Earlier batch pipeline](../analysis/auditory-legacy.md)

## Analysis

??? question "A script stops with FileNotFoundError on a file under BATCH_ANALYSIS"
    It reads the tables of a script that has not been run yet. Run the scripts in order. [Reproduce the thesis](../analysis/reproduce-thesis.md#the-order-to-run-them-in)

??? question "A mixed model script cannot find R"
    The scripts look for `Rscript` on the PATH, then in the newest R under `C:/Program Files/R`. Install R 4 with `lme4`, `lmerTest` and `emmeans`. [Thesis scripts](../analysis/auditory.md#what-you-need)

??? question "The Bayesian fits of Experiment 2 crawl"
    PyTensor needs a C compiler on the PATH, or it runs in slow Python mode. [Value model quick guide](../analysis/model-validation.md#5-how-to-run-it)

??? question "`exp1_complexity.py` skips its second part"
    It could not find the vocalisation recordings. Add your folder to `VOCALISATION_FOLDERS` at the top of the script. [Thesis scripts](../analysis/auditory.md#experiment-1-eight-arms-six-stimulus-sets)

??? question "`run_pipeline.py` stops with `missing script ...analysis\analysis\...`"
    The runner looks one folder too deep. Run the pose stages one by one. [Pose pipeline](../analysis/pose.md#quick-start)

??? question "`No module named 'sleap_nn'`, or `'sleap' is not installed`"
    The wrong Python. Run the pose stages with the sleap-nn interpreter by its full path. [Pose pipeline](../analysis/pose.md#what-you-need)

??? question "Cropping fails for a whole mouse or cohort"
    Those videos come from another rig. Calibrate it and run again with `--calibration_dir`. [Cropping and aligning](../analysis/crop-align.md#troubleshooting)

??? question "The tactile scripts find no sessions"
    Check `MOUSE_ID` and `BASE_PATH` in `config.py`; the scripts look in `<BASE_PATH>/mouse <MOUSE_ID>/`. [Tactile pipeline](../analysis/tactile.md#troubleshooting)

??? question "An overlay video will not play"
    It uses the `mp4v` codec. Open it in VLC.

## Documentation

??? question "`mkdocs serve` fails, or the site looks unstyled"
    Install the documentation extra from the repository root: `pip install -e ".[docs]"`. Check the whole site with `mkdocs build --strict`, which the documentation workflow also runs.
