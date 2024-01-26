import pandas as pd

file=pd.read_csv('J1644-4559.tim', skiprows=1, comment='C', sep=' ', usecols=[0,1,2,3,4], skipinitialspace=True, names=['Name', 'Freq','MJD','ToA','Antenna'])

file_v2 = file.sort_values(by=["MJD"])

delete = []

for i in range(len(file_v2)-1):
     if file_v2["MJD"][i+1]==file_v2["MJD"][i]:
         delete.append(i)

file_v3=file_v2.drop(delete, axis=0)

new_row=pd.DataFrame({"Name" : 'FORMAT 1'}, index=[0]) 
#Luego sacarle las comillas
file_v3= pd.concat([new_row, file_v3.loc[:]]).reset_index(drop=True)

file_v3.to_csv("J1644-4559_clean.tim", sep=' ',header=False, index=False, index_label=None)


