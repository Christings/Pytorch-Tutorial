import pandas as pd

data=pd.read_csv('students.csv')
print(data)

data.to_pickle('student.pickle')