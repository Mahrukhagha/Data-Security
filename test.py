
import acro 

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from acro import ACRO, add_constant
from pandas.io.stata import StataReader
pd.set_option('display.max_columns', None)  # Show all columns


acro = ACRO()
data = pd.read_stata("ZA5282_v1-0-0.dta", convert_categoricals=False) 
print(data)
print(f" german column entries in raw file {data.german.unique()}")
table = pd.crosstab(data.german, data.eastwest)
#print(table)
safe_table = acro.crosstab(
    data.german, data.eastwest, rownames=["german"], colnames=["eastwest"]
)
print(safe_table)

safe_table2 = acro.crosstab(data.eastwest, data.german, values=data.eastwest, aggfunc="mean")
print(safe_table2)


table3 = acro.pivot_table(
    data, index=["eastwest"], values=["german"], aggfunc=["mean", "std"]
)
print(table3)


data["eastwest"].replace(
    to_replace={
        "not_recom": "0",
        "recommend": "1",
        "very_recom": "2",
        "priority": "3",
        "spec_prior": "4",
    },
    inplace=True,
)
data["eastwest"] = pd.to_numeric(data["eastwest"])

new_df = data[["eastwest", "german"]]
new_df = new_df.dropna()

y = new_df["eastwest"]
x = new_df["german"]
x = add_constant(x)

results = acro.ols(y, x)
results.summary()