from eli5.permutation_importance import get_score_importances
from f_DGBCox import DGBCox
import numpy as np
import pandas as pd
from tfdeepsurv.datasets import survival_df

train_data = pd.read_csv("data/METABRIC1980.csv")
clindex = train_data.columns
colname_e = 'event'
colname_t = 'time'
surv_train = survival_df(train_data, t_col=colname_t, e_col=colname_e, label_col="Y")

input_nodes = len(surv_train.columns) - 1
input_samples = len(surv_train.index)
hidden_layers_nodes = [int(input_nodes / 8), int(input_nodes / 16), int(input_nodes / 64)]

nn_config = {
        "learning_rate": 0.001,
        "learning_rate_decay": 0,
        "activation": 'tanh',
        "L1_reg": 1e-4,
        "L2_reg": 1e-2,
        "Lamda_3": 1,
        "Lamda_4": 5,
        "Lamda_5": 0.01,
        "optimizer": 'adam',
        "seed": 1
    }

model = DGBCox(
        hidden_layers_nodes,
        input_samples,
        input_nodes,
        nn_config
    )

model.build_graph()

Y_col = ["Y"]
X_cols = [c for c in surv_train.columns if c not in Y_col]

def score(X, y):

    data = np.c_[X, y]
    data = pd.DataFrame(data, columns =clindex)
    data = survival_df(data, t_col=colname_t, e_col=colname_e, label_col="Y")

    ci = model.evals(data[X_cols], data[Y_col], load_model = "model/DGBCox_final.ckpt")
    return ci

def VIP():
    # ... load data, define score function
    dn = np.array(["transbig", "unt", "upp", "mainz", "nki", "GSE6532", "GEO", "TCGA753", "TCGA500",
                   "UK", "HEL", "TCGA1093"])
    for i in range(12):
        ddata = pd.read_csv("data/" + dn[i] + ".csv")
        ddata = ddata.to_numpy("float32")
        n, p = ddata.shape
        X_in = ddata[:, :-2]
        Y_in = ddata[:, (p-2):p]

        base_score, score_decreases = get_score_importances(score, X_in, Y_in)

        if(i == 0):
            feature_importances = np.mean(score_decreases, axis=0)
        else:
            feature_importances = np.vstack((feature_importances,np.mean(score_decreases, axis=0)))

    np.savetxt("vip.csv", feature_importances, delimiter=",")
    print("OK")

if __name__ == '__main__':
    VIP()
