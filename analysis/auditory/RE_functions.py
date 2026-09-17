
import os
import numpy as np 
import pandas as pd 


def get_REs(RE_folder_path, time): 
    """
    input: 
        time in ms 
        RE_folder_path to csv files with 10 cols: ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'ROI7', 'ROI8', 'Time', 'Entrance']
        # each row is a frame, 1 for in ROI, 0 if not 
    """

    def get_ROI_frames(RE_folder_path, time):
        """
        for each mouse, session, roi, 1st entrance --> time, count number of frames mouse is in each
        output:
            df with mouse, day, num frames occupied each ROI
        """
        # 30 fps 
        # 30/1000 frames / ms 
        # time in ms * frames per ms = frame num 
        time_in_frames = time * (30/ 1000)
        
        refiles = os.listdir(RE_folder_path)
        
        ROI_frames_list = []
        
        for f in refiles: 
            path = os.path.join(RE_folder_path, f)
            ROI_csv = pd.read_csv(path)
            #note time in bellow means how many frames 
            ROI_csv.columns = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'ROI7', 'ROI8', 'Time', 'Entrance']
            
            ## get row of first entrance, trial start (frame number of first entrance to maze)
            trial_start_time = ROI_csv.loc[(ROI_csv['Entrance'] == 1)].head(1)['Time'].item()
            ## TODO 
            trial_end_time = trial_start_time + time_in_frames
        
            ##get trial rows 
            trial_rows = ROI_csv.loc[(ROI_csv['Time'] >= trial_start_time) &
                                         (ROI_csv['Time'] <= trial_end_time)]
            
            ROI_frames = pd.DataFrame({'ROI1': [None], 'ROI2': [None], 'ROI3': [None], 'ROI4': [None], 'ROI5': [None], 'ROI6': [None],
                                     'ROI7': [None], 'ROI8': [None]})
            
            ROIs = ['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'ROI7', 'ROI8']
            
            ## count the number of frames where mouse is in each roi 
            for r in ROIs: 
                ROI_frames.loc[0, r] = len(trial_rows.loc[(trial_rows[r] == 1)])
                
            ROI_frames['mouse'] = [f[0:5]]
            ROI_frames['day'] = [f[0:-13][5::]]
            
            ROI_frames_list.append(ROI_frames) 
        
        
        ROI_frames_df = pd.concat(ROI_frames_list).reset_index(drop = True)
        return ROI_frames_df

    def frames_to_probability(ROI_frames_df): 
        """
        output: 
            RE_df --> RE per mouse, session 
            probability_df --> probability of finding mouse in ROI if is in an ROI, for plotting 
        """
        #entropy_lambda = lambda prop, n: -np.sum(prop[prop > 0] * np.log2(prop[prop > 0]))/ np.log2(n)
        entropy_lambda = lambda prop, n: -np.sum(prop * np.log2(prop))/ np.log2(n)
        
        #total number of frames occupying any ROI 
        total_frames = ROI_frames_df.iloc[:, 0:-2].sum(axis = 1)
        
        ROI_probability_df = np.divide(np.array(ROI_frames_df.iloc[:, 0:-2]), np.array(total_frames).reshape(-1, 1))
        
        REs = []
        
        ## TODO --> remove for loop later  
        for r in range(len(ROI_probability_df)): 
            row = ROI_probability_df[r]
            # remove 0's 
            # and turn to float , np.log2 needs float input
            row = row[row > 0].astype(float)
            RE = entropy_lambda(row, 8)
            REs.append(RE)
        
        RE_df = pd.DataFrame({'RE': REs, 'mouse': ROI_frames_df['mouse'], 'day': ROI_frames_df['day']})
        ########
        ROI_probability_df = pd.DataFrame(ROI_probability_df)
        ROI_probability_df['mouse'] = ROI_frames_df['mouse']
        ROI_probability_df['day'] = ROI_frames_df['day']
        ########
        return RE_df, ROI_probability_df
    
    ROI_frames_df = get_ROI_frames(RE_folder_path, time)
    RE_df, ROI_probability_df = frames_to_probability(ROI_frames_df)
    
    return RE_df, ROI_probability_df
