"""
Created on Sat Dec 18 12:53:31 2021

@author: abdulatif albaseer
"""


import pandas as pd
import pathlib2 as pl2
head=['Delivery Date',	'Generator',	'Fuel Type',	'Measurement',	'Hour 1',	'Hour 2',	'Hour 3',	'Hour 4',	'Hour 5',	'Hour 6',	'Hour 7',	'Hour 8',	'Hour 9',	'Hour 10',	'Hour 11',	'Hour 12',	'Hour 13',	'Hour 14',	'Hour 15',	'Hour 16',	'Hour 17',	'Hour 18',	'Hour 19',	'Hour 20',	'Hour 21','Hour 22',	'Hour 23',	'Hour 24']
"Data loader"
def df_concat(path):
  ps = pl2.Path(path)
  dfs = (
      pd.read_csv(p, skiprows=4, header=None, encoding='utf8' ).iloc[0:,:-1].set_axis(head, axis=1) for p in ps.glob('*.csv')
  )
  res20 = pd.concat(dfs)
  return res20
