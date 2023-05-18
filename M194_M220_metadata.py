import pandas as pd

num_samples = 12
mice = ["MMM1","MMF2", "MMF3", "STM1", "STM2", "STF3", 'MMM4', 'MMF5', 'STF4', 'STM5', 'STF6', 'STM7']
species = ["Mmus"]*3 + ["Steg"]*3 + ["Mmus"]*2 + ["Steg"]*4
sex = ["male", "female", "female", "male", "male", "female", "male", "female", "female", "male", "female", "male"]
dataset = ['M194']*6 + ['M220']*6
colors = {'M194':'blue', 'M220':'orange'}
metadata = pd.DataFrame({'mice':mice, "species":species, "sex":sex, "dataset":dataset})