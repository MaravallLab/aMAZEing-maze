# Welcome to the tactile experiment component of the aMAZEing maze

## Configuration files

`simplerCode.py` needs two CSVs that sit next to it (they are resolved relative
to the script, so you can launch it from any directory):

- **`grating_maps.csv`** — index column is the reward location (`A`–`D`); every
  other column is named `motor <servo>` and holds the `<servo> <angle>` string
  sent to the Arduino as `grt<servo> <angle>` (servo names match
  `firmware/arduino/servo_control/servo_control.ino`: `L`, `R`, `LL`, `LR`,
  `RL`, `RR`, …). At the start of every trial all servos are first sent to
  angle 0, then the row for the trial's reward location is applied.
- **`reward_sequences.csv`** — columns `sessionID` (`Stage 1`…`Stage 4`),
  `rewloc`, `portprob` (fraction of the session's trials at that port),
  `rewprob` (probability a trial at that port is rewarded), `wrongallowed`.
  `create_trials()` in `supFun.py` samples the trial list from this table.

The files in the repo are **templates** reconstructed from what the code expects.
Replace them with the values used on your rig before collecting data.

ROIs (`entrance1`, `entrance2`, `rewA`–`rewD`) are drawn interactively on the
first run and saved as `rois1.csv` in the recordings folder
(`~/Desktop/maze_recordings`); set `drawRois = True` to redraw them.



For posthoc analysis, we are currently relying on DeepLabCut output, which is not particularly robust. 
We are working on data labeling for robustness, and will compare performance of [Yolov8](https://yolov8.com/), [STPoseNet](https://github.com/lvrgb777/STPoseNet/tree/master) (check out their [paper](https://www.sciencedirect.com/science/article/pii/S2589004224009945)!), and [DLC](https://github.com/DeepLabCut/DeepLabCut) (check out their [paper](https://www.nature.com/articles/s41596-019-0176-0)!) for accuracy. 

**For newer versions of the script** , <a href="simplermaze.py">simplermaze.py</a> outputs the video of the full session, keeping track of the start and end frames for each trial in the output session data csv file. Running <a href="post_process_session.py">post_process_session.py</a> after the trial is completed, will create a subdirectory containing the segments of the individual trials, that together with the summary of the trial segments and frames ranges, can then be used to extrapolate the trajectories per trials with the chosen keypoint estimation tool

**For older versions of the script/data**, the scripts in <a href="/analysis/trials_segmentation/">trials_segmentation</a> will handle the segmentation by reprocessing the videos. 

There are different ways to **plot this data**, right now we are using <a href="/analysis/first_paper_exploratory_analysis/make_3d_dlc_plot_trajectories.py">make_3d_dlc_plot_trajectories.py</a>. This might change or we might make a new version when the newer versions of the script will be running , but that's a job for future us, isn't it?

Now, important. To calculate the speed, we need to convert the euclidean distance calculated in pixels per frame, to cm/s. The script should handle this automatically as long as you enter the **px/cm** as one of the arguments. To find out how many px/cm, please run <a href="/analysis/first_paper_exploratory_analysis/measure_pixel_per_cm.py">measure_pixel_per_cm.py</a>. 