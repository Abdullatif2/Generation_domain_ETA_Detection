
"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""


import matplotlib.pyplot as plt

def df_visualization(df):
  df.boxplot(column=['Hour 15'],by=['month'])
  plt.show()
  month=df.groupby(df['month']).mean()
  valx={"Hour 1":      "1",
  "Hour 2":      "2",
  "Hour 3":     "3",
  "Hour 4":      "4",
  "Hour 5":      "5",
  "Hour 6":      "6",
  "Hour 7":     "7",
  "Hour 8":      "8",
  "Hour 9":     "9",
  "Hour 10":    "10",
  "Hour 11":    "11",
  "Hour 12":    "12",
  "Hour 13":    "13",
  "Hour 14":    "14",
  "Hour 15":    "15",
  "Hour 16":    "16",
  "Hour 17":    "17",
  "Hour 18":     "18",
  "Hour 19":     "19",
  "Hour 20":     "20",
  "Hour 21":     "21",
  "Hour 22":     "22",
  "Hour 23":     "23",
  "Hour 24":     "24"}
  month=month.rename(columns=valx)
  for index, row in month.iloc[:,0:-1].iterrows():
      plt.plot(row)
  plt.xlabel('Hours per day')
  plt.ylabel('Generated power')
  
  # displaying the title
  plt.title("Mean generated power per month")
  plt.show()
  day=df.groupby(df['Dayofweek']).mean()
  day=day.rename(columns=valx)
  for index, row in day.iloc[:,0:-1].iterrows():
      plt.plot(row)
  plt.xlabel('Hours per day')
  plt.ylabel('Generated power')
  
  # displaying the title
  plt.title("Mean generated power per dayof week")
  plt.show()