
import numpy as np 
import pandas as pd 
import os 


def list_files_oi_paths(file_path, search_terms = [".mp4"], search_all = True, search_to_day = 2, search_to_mouse = 2):
    """
    
    returns file paths to files of interest 

    Inputs: 
        search_terms = ["a", "b"] --> searches for files with a and b in filename 
        search_all = True --> seach through all mouse and day subfolders 
        search_to_day = how many folders in days to search through 
        search_to_mouse = how many mouse folders to search through 
    Outputs: 
        files_oi_paths = data frame with mouseID, day and path to file that matches search terms 
    """
    
    if search_all: 
        search_to_mouse = None
        search_to_day = None
    
    files_oi_paths = []  
    
    #assumes is days folders at file_path locaiton 
    days_folders = os.listdir(file_path)
    
    for day in days_folders[0:search_to_day]: 
        
        # skip single files  
        if os.path.isfile(os.path.join(file_path, day)):
            print('is file not folder')
            continue
        
        mice_folders = os.listdir(os.path.join(file_path, day))
        
        for n in mice_folders[0:search_to_mouse]: 
            mouse_id = n[-5:]
            # skip single files  
            if os.path.isfile(os.path.join(file_path, day, n)):
                print('is file not folder')
                continue
            
            #assumes is mouse folders inside days_folders 
            files_in = os.listdir(os.path.join(file_path, day, n))
            
            file_oi = [f for f in files_in if all(search_term in f for search_term in search_terms)]
            
            # check if missing files or duplicates 
            assert len(file_oi) == 1, f"Missing / >1 file for {n}"
                 
            add = pd.DataFrame({'day': [day]* len(file_oi), 'mouse': [mouse_id]*len(file_oi), 'path': [os.path.join(file_path, day, n, f) for f in file_oi]})
            files_oi_paths.append(add)
            
    files_oi_paths = pd.concat(files_oi_paths)
        
    return files_oi_paths

def get_mouse_info(file_path, unique_mice):
    """
    """
    def concat_path_dfs(paths_df):
        concat_df = []
        for r in range(len(paths_df)): 
            path = paths_df.iloc[r]['path']
            df = pd.read_csv(path)
            concat_df.append(df)
        concat_df = pd.concat(concat_df)
        return concat_df
    
    paths = []
    # search all days because is possible for mouse to not enter maze on day 1 
    mouse_info_file_paths = list_files_oi_paths(file_path = file_path, search_terms = [".csv", "mouse", "time"] , search_all = True)
    for u in unique_mice: 
        row = mouse_info_file_paths.loc[(mouse_info_file_paths['mouse'] == u)].iloc[0]['path']
        paths.append(pd.DataFrame({'path' : [row]}))
    paths = pd.concat(paths)
    mouse_info = concat_path_dfs(paths)
    mouse_info['mouse'] = [n[-5:] for n in mouse_info['animal ID']] 
    return mouse_info



def get_session_trial_info(trials_tables, search_all = True, search_days = ["w1_d1"]):

    """
    inputs: 
        trials_table --> list of csv paths, mouse IDs and days 
        (list_files_oi_paths(search_terms = ["trials", "time", '.csv'], search_all = True))
        search_all --> set to TRUE for all days 
    outputs: 
            df for per trial information for each mouse and day (sound/ silent visits , sound/silent time, time in maze)
    """

    def sound_silent_rows(rows, day):
        # must match terms used in maze csvs 
        match day:
            case "w1_d1":
                silent_rows = rows.loc[(rows['frequency'] == "silent_arm")]
                sound_rows = rows.loc[(rows['sound_type'] != "control")]
                vocalisation_rows = rows.loc[(rows['frequency'] == "vocalisation")]
            case "w1_d2" | "w1_d3":
                silent_rows = rows.loc[(rows['frequency'] == "0")]
                sound_rows = rows.loc[(rows['frequency'] != "0")&
                                      (rows['frequency'] != "vocalisation")]
                vocalisation_rows = rows.loc[(rows['frequency'] == "vocalisation")]
            case "w2_vocalisations":
                silent_rows = rows.loc[(rows['frequency'] == "0")]
                sound_rows = rows.loc[(rows['frequency'] != "0")&
                                      (rows['frequency'] != "vocalisation")]
                vocalisation_rows = sound_rows
            case "w1_d4":
                ######## 
                sound_rows = rows 
                silent_rows = np.nan
                vocalisation_rows = np.nan
                #######
            case "w2_sequences":
                silent_rows = rows.loc[(rows['pattern'] == "silence")]
                sound_rows = rows.loc[(rows['pattern'] != "silence")&
                                      (rows['pattern'] != "vocalisation")]
                vocalisation_rows = rows.loc[(rows['pattern'] == "vocalisation")]
        return sound_rows, silent_rows, vocalisation_rows
                
       
    if search_all: 
        search_days = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
    trials_tables = trials_tables.loc[(trials_tables['day'].isin(search_days))]
    
    
    per_trial_df = []
    for p in range(len(trials_tables)):
        df = pd.read_csv(trials_tables['path'].iloc[p]).fillna(0)
        mouse = trials_tables['mouse'].iloc[p]
        day = trials_tables['day'].iloc[p]
        
        #ignore habituation and silent trials with no sounds 
        trials = [f for f in df['trial_ID'].unique() if f % 2 == 0]
        for t in trials:
            
            rows = df.loc[(df['trial_ID'] == t)]
            
            total_roi_time = rows['time_spent'].sum()
            sound_rows, silent_rows, vocalisation_rows = sound_silent_rows(rows, day)
            
            sound_visits = sound_rows['visitation_count'].sum()
            sound_time = sound_rows['time_spent'].sum()
            
            # No silent/vocalisation arm in w1_d4
            if day == "w1_d4":
                silent_visits = np.nan
                silent_time = np.nan
                vocalisation_time = np.nan
                vocalisation_visits = np.nan
            else: 
                silent_visits = silent_rows['visitation_count'].sum()
                silent_time = silent_rows['time_spent'].sum()
                vocalisation_visits = vocalisation_rows['visitation_count'].sum()
                vocalisation_time = vocalisation_rows['time_spent'].sum()
                
            to_add = ( 
                pd.DataFrame({'mouse':[mouse],'day':[day],'trial':[t], 'sound_visits':[sound_visits],
                              'sound_time':[sound_time], 'silent_visits':[silent_visits],
                              'silent_time': [silent_time], 'total_roi_time': [total_roi_time],
                              'vocalisation_time': vocalisation_time, 'vocalisation_visits': vocalisation_visits})
                )
            per_trial_df.append(to_add)
    per_trial_df = pd.concat(per_trial_df)
    return per_trial_df


def add_days_column(df): 
    to_day = {'w1_d1': 1,
              'w1_d2': 2, 
              'w1_d3': 3, 
              'w1_d4': 4, 
              'w2_sequences': 5,
              'w2_vocalisations': 6}
    df['day_number'] = df['day'].map(to_day)
    return df


def add_cohort_column(df):
    ## CHECK BELLOW WITH ALEJANDRA 
    cohort = {
         "29/03/2025" : 'A',
         "09/05/2025" : 'B',
         "2/8/2025": 'C',
         "02/08/2025" : 'D',
         "02/09/2025": 'E',
         "23/08/2025": 'F'
         }
    df['cohort'] = df['animal birth date'].map(cohort)
    return df


def get_habituation_info(trials_tables, mouses, search_all = True, search_days = ["w1_d1", "w1_d2", "w1_d3", "w1_d4"]):
    
    if search_all: 
        search_days = ["w1_d1", "w1_d2", "w1_d3", "w1_d4", "w2_sequences", "w2_vocalisations"]
    
    trials_tables = trials_tables.loc[(trials_tables['day'].isin(search_days))]
    
    per_trial_df = []
    for p in range(len(trials_tables)):
        df = pd.read_csv(trials_tables['path'].iloc[p]).fillna(0)
        mouse = trials_tables['mouse'].iloc[p]
        day = trials_tables['day'].iloc[p]
        
        rows = df.loc[(df['trial_ID'] == 1)]
 
        total_ROI_time = rows['time_spent'].sum()
        total_ROI_visitations = rows['visitation_count'].sum()
            
        to_add = ( 
            pd.DataFrame({'mouse':[mouse],'day':[day], 'habit_total_ROI_time':[total_ROI_time],
                          'habit_total_ROI_visitations':[total_ROI_visitations]})
            )
        per_trial_df.append(to_add)
    per_trial_df = pd.concat(per_trial_df)
    return per_trial_df


def avg_var(df, var):
    """
    inputs: 
        df --> session wise df 
        var --> var in session wise df you want mouse wise mean for 
    ## TODO --> change to add new column instead of replacing 
    """
    dff = df[['mouse', var]]
    dff = dff.groupby(['mouse']).mean().reset_index()
    dff = dff.rename(columns = {var : f'mouse_AVG_{var}'})
    return dff

def mean_subtract(df, var):
    """
    adds mean subtracted var column 
    """
    dff = df.copy()
    on_means = dff[["mouse", var]].groupby(["mouse"]).mean().reset_index()
    on_means = on_means.rename(columns = {var : "MEAN"})
    dff = dff.merge(on_means, on = "mouse")
    dff[f"{var}_mean_subtracted"] = dff[var] - dff["MEAN"]
    dff = dff.drop(columns = ["MEAN"])
    return dff
