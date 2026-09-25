import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate


def kfold_model(model, X, y, cv=5):
    return cross_validate(
        model,
        X,
        y.values.ravel(),  # 1D, otherwise it would be a column vector and it would get a warning while training the NN.
        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        return_train_score=True,
        return_estimator=True,
        # cv=rkf,
        cv=cv,
    )


def get_kfold_results(
    scores: dict[str, pd.DataFrame], sort_by: str = "r2", show_train: bool = False
):
    return (
        pd.DataFrame(
            {
                name: {
                    col: s[col].mean()
                    for col in s.columns
                    if col.startswith(("test_",) + (("train_",) if show_train else ()))
                }
                for name, s in scores.items()
            }
        )
        .T.round(3)
        .sort_values(f"test_{sort_by}", ascending=False)
        .rename(
            columns={
                "test_r2": "R-squared test",
                "train_r2": "R-squared train",
                "test_neg_root_mean_squared_error": "RMSE test",
                "train_neg_root_mean_squared_error": "RMSE train",
                "test_neg_mean_absolute_error": "MAE test",
                "train_neg_mean_absolute_error": "MAE train",
            }
        )
    )


def show_correlation_matrix(corr: pd.DataFrame, title: str) -> None:
    # Show only the lower triangle since to avoid visual clutter
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
