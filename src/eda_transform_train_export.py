from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml

def load_config()-> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.

    Reads the 'config.yml' file from the current directory
    and returns its contents as a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing configuration keys and values.
    """    
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Args:
        path (str): The path or URL to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def print_df(df):
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    return None


def clean_required_rows(df: pd.DataFrame, required_fields: list[str]) -> pd.DataFrame:
    """
    Remove rows that have NaN in any of the required fields.
    """
    return df.dropna(subset=required_fields)

def fill_missing_continuous(df: pd.DataFrame, continuous_features:List[str]) -> pd.DataFrame:
    # For ease of handling, missing continuous values are filled with their respective column means.
    df.fillna(df.mean(numeric_only=True),inplace=True)
    return df

def remove_columns(df: pd.DataFrame, continuous_features) -> pd.DataFrame:
    df = df.drop(columns=continuous_features)
    return df


def clean_primary_col(df, primary_field) -> pd.DataFrame:
    df = clean_required_rows(df, primary_field)
    df = df.drop_duplicates(subset=primary_field, keep="first")
    return df


def one_hot_encode_columns(df: pd.DataFrame, nominal_cols: List[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on one or more specified nominal (categorical) columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing nominal categorical columns.
        nominal_cols (List[str]): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns replacing the original ones.
    """
    for col in nominal_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols)
    return df

def encode_ordinal_features(df: pd.DataFrame, ordinal_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Apply ordinal encoding to specified columns using provided category orderings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ordinal_map (Dict[str, List[str]]): A dictionary where each key is a column name
                                            and its value is a list of ordered categories
                                            (from lowest to highest).

    Returns:
        pd.DataFrame: DataFrame with ordinal features encoded as integers.
    """
    for col, categories in ordinal_map.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes

    return df


def data_preprocessing(df: pd.DataFrame, features) -> pd.DataFrame:
    """
    Clean the raw dataset by applying filtering and imputation steps.

    Args:
        df (pd.DataFrame): Raw input DataFrame.
        features (dict): Feature configuration dictionary.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = clean_primary_col(df, features["primary_field"])
    df = clean_required_rows(df, features["required_fields"])
    df = fill_missing_continuous(df, features["continuous"])
    return df


def feature_engineering(df: pd.DataFrame, features) -> pd.DataFrame:
    """
    Perform feature transformations and target generation.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        features (dict): Feature configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df = one_hot_encode_columns(df, features["nominal"])
    df = encode_ordinal_features(df, features["ordinal"])
    
    # Mock a target column from task_status
    df[features["target"]] = df["task_status"].apply(lambda x: True if x in ["waiting", "running"] else False)

    df = remove_columns(df, features["columns_to_drop"])
    return df

def split_dataset(
    df: pd.DataFrame,
    split_config: Dict[str, float],
    stratify_col: str = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The full dataset to split.
        split_config (Dict[str, float]): Dictionary with keys 'train', 'val', and 'test'.
        stratify_col (str, optional): Column name to stratify on. Default is None.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    train_size = split_config["train"]
    val_size = split_config["val"]
    test_size = split_config["test"]

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    stratify = df[stratify_col] if stratify_col else None

    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=stratify, random_state=random_state
    )

    stratify_temp = temp_df[stratify_col] if stratify_col else None

    val_ratio = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, stratify=stratify_temp, random_state=random_state
    )

    return train_df, val_df, test_df

def main() -> None:
    # Load config
    config = load_config()

    # Load Data
    dataset_url = config["dataset"]["url"]
    df = load_data(dataset_url)

    # Perform data Pre-processing & Feature Engineering
    features = config["features"]
    df = data_preprocessing(df, features)

    df = feature_engineering(df, features)
    

    split_cfg = config["split"]
    train_df, val_df, test_df = \
        split_dataset(df, split_cfg, stratify_col=config["features"]["target"])
    
    

if __name__ == "__main__":
    main()
