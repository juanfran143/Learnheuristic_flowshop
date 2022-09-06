import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def box_plot_results_modify_good():
    df = pd.read_excel("output/Results2.xlsx")
    df.loc[df["ML_used"], "ML_used"] = "Learnheuristic"
    df.loc[df["ML_used"] == False, "ML_used"] = "Deterministic"

    instances = np.unique(df.loc[:,"instance_name"])
    for ins in instances:
        df_3 = df.loc[df.loc[:, "instance_name"] == ins, :]


        ax = sns.boxplot(x="h", y="makespan_bb", hue="ML_used", data=df_3, palette="Set3",
                         showmeans=True, meanprops={"marker": "o",
                                                    "markerfacecolor": "white",
                                                    "markeredgecolor": "black",
                                                    "markersize": "5"})


        #ax.axhline(0, ls='--', color='r')

        plt.ylabel("Black box makespan")
        plt.xlabel("Deterministic factor")
        plt.title("Comparation deterministic vs learnheuristic")
        plt.show()



