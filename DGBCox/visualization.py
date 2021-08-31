import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

def plot_surv_curve(df_survf, title="Survival Curve", save_file=''):
    """
    Plot survival curve.
    Parameters
    ----------
    df_survf: DataFrame or numpy.ndarray
        Survival function of samples, shape of which is (n, #Time_Points).
        `Time_Points` indicates the time point presented in columns of DataFrame.
    title: str
        Title of figure.
    save_model: string
            Path for saving model.
    """
    f = plt.figure()
    if isinstance(df_survf, DataFrame):
        plt.plot(df_survf.columns.values, np.transpose(df_survf.values))
    elif isinstance(df_survf, np.ndarray):
        plt.plot(np.array([i for i in range(df_survf.shape[1])]), np.transpose(df_survf))
    else:
        raise TypeError("Type of arguement is not supported.")

    plt.title(title)
    #plt.show()
    f.savefig(save_file + '.pdf', bbox_inches='tight')

def plot_km_survf(data, t_col="t", e_col="e", save_file=''):
    """
    Plot KM survival function curves.
    Parameters
    ----------
    data: pandas.DataFrame
        Survival data to plot.
    t_col: str
        Column name in data indicating time.
    e_col: str
        Column name in data indicating events or status.
    save_model: string
            Path for saving model.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    f = plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    kmfh = KaplanMeierFitter()
    kmfh.fit(data[t_col], event_observed=data[e_col], label="KM Survival Curve")
    kmfh.survival_function_.plot(ax=ax)
    plt.ylim(0, 1.01)
    plt.xlabel("Time")
    plt.ylabel("Probalities")
    plt.legend(loc="best")
    add_at_risk_counts(kmfh, ax=ax)
    #plt.show()
    f.savefig(save_file + '.pdf', bbox_inches='tight')