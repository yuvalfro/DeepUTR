import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import AnchoredText

L = 110
SEQ_LST = ["TenT", "AUrich", "PUM", "PolyU", "miR430"]
SEQ_NUCS = ["TTTTTTTTTT", "TATTATTTAT", "TATGTAAATATGTA", "TTTTTTTTTTTTT", "GCACTT"]


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]

def get_data(statistics_csv, prediction_csv):

    statistics_list = pd.read_csv(statistics_csv).values.tolist()
    for lunp in statistics_list:
        del lunp[0:29]  # Delete index+28 first elements
        del lunp[110:138]  # Delete last 28 elements according to new index after delete the first 28
    df_statistics = pd.DataFrame(statistics_list)

    df_prediction = pd.read_csv(prediction_csv)
    return df_statistics, df_prediction

for seq,nucs in zip(SEQ_LST,SEQ_NUCS):
    statistics_csv = f"../DeepUTR/lunps-results/lunps_results_{seq}.csv"
    prediction_csv = f"../DeepUTR/files/predictions_model_8_points_id_1_dynamics_-{seq}.csv"
    df_statistics, df_prediction = get_data(statistics_csv, prediction_csv)
    n = int((L-len(nucs))/2)
    ##### Calculate pearson correlation
    avg_prob = df_statistics.iloc[:, n:n+len(nucs)].mean(axis=1)
    deg_rate = df_prediction['8h']
    data = {f"{seq} AVG": avg_prob, "Deg Rate": deg_rate}
    df = pd.concat(data, axis=1)
    corr = df.corr(method='pearson')
    pval = df.corr(method=pearsonr_pval)
    print(f"Pearson correlation coefficient is:\n {corr}")
    print(f"\np-value is:\n {pval}")

    xy = np.vstack([avg_prob, deg_rate])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    avg_prob, deg_rate, z = avg_prob[idx], deg_rate[idx], z[idx]
    fig, ax = plt.subplots()
    sc = ax.scatter(avg_prob, deg_rate, c=z, s=10)
    ax.set_title(f"Correlation between base-pairing probabilities\n to mRNA degradation rate\nSequence {seq}: {nucs}")
    cbar = fig.colorbar(sc)
    cbar.set_label("Number of points", loc='center')
    anchored_text = AnchoredText(f"\u03C1 = {round(corr.values[0][1],4)}\np-value = {format(pval.values[0][1], '.4g')}", loc=2)
    ax.add_artist(anchored_text)
    ax.set_xlabel(f"{seq} base-pairing probabilities average")
    ax.set_ylabel("mRNA degradation rate")
    plt.savefig(f"correlation_{seq}.jpg", bbox_inches='tight')