from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_binary_split(
    meta_df: pd.DataFrame,
    *,
    target_col: str = "target",
    group_col: str = "src_file",
    split_col: str = "is_train",
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    output_df = meta_df.copy()
    output_df[split_col] = True
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for _, val_idx in splitter.split(output_df, output_df[target_col], output_df[group_col]):
        output_df.iloc[val_idx, output_df.columns.get_loc(split_col)] = False
        break
    return output_df


def assign_folds(
    train_df: pd.DataFrame,
    *,
    target_col: str = "target",
    group_col: str = "src_file",
    fold_col: str = "fold",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    output_df = train_df.copy()
    output_df[fold_col] = None
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (_, val_idx) in enumerate(splitter.split(output_df, output_df[target_col], output_df[group_col])):
        output_df.iloc[val_idx, output_df.columns.get_loc(fold_col)] = fold_id
    return output_df
