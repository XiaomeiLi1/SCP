from tfdeepsurv.datasets import survival_df
from f_DGBCox import *
import sys

if __name__ == "__main__":

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
    # specify the colnames of observed status and time in your dataset
    colname_e = 'event'
    colname_t = 'time'

    # survival data must be transformed by the functionality tfdeepsurv.datasets.survival_df
    # Y =  time, if event = 1
    # Y = -time, if event = 0

    # Number of features and samples in your dataset
    input_nodes = 7913
    input_samples = 1980
    # Specify your neural network structure.
    hidden_layers_nodes = [int(input_nodes / 8), int(input_nodes / 16), int(input_nodes / 64)]

    # ESSENTIAL STEP-1: Pass arguments
    # the arguments of DGBCox are obtained by Bayesian Hyperparameters Tuning.
    model = DGBCox(
        hidden_layers_nodes,
        input_samples,
        input_nodes,
        nn_config
    )

    # ESSENTIAL STEP-2: Build Computation Graph
    model.build_graph()

    # model saving and loading is also supported.
    # read comments of `train()` function if necessary.

    # ESSENTIAL STEP-3: Prediction
    dn = sys.argv[1:]

    ci_list = np.zeros([len(dn), 1])

    for i in range(len(dn)):
        test_data = pd.read_csv("data/" + dn[i] + ".csv")
        surv_test = survival_df(test_data, t_col=colname_t, e_col=colname_e, label_col="Y")
        Y_col = ["Y"]
        X_cols = [c for c in surv_test.columns if c not in Y_col]
        ci_list[i] = model.evals(surv_test[X_cols], surv_test[Y_col], load_model="model/DGBCox_final.ckpt")

    print(ci_list)
