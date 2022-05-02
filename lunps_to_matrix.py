import os
import numpy as np
import pandas as pd

lunps_folder = "lunp_files"

matrix = np.zeros((58, 166))
for i in range(1,59):
    f = os.path.join(lunps_folder, f"{i}_lunp")
    # checking if it is a file
    if os.path.isfile(f):
        file = open(f)
        Lines = file.readlines()
        count = 0
        for line in Lines:
            if count == 0 or count == 1:
                count += 1
                continue
            value = line.split('\t')[1][:-1]
            column = line.split('\t')[0]
            matrix[i-1, int(column) - 1] = np.squeeze(value)

pd.DataFrame(matrix).to_csv("../DeepUTR/lunps_results.csv")