# Standard library imports
from typing import Dict, Any, List, Tuple, Union

# Third-party library imports
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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


def clean_required_rows(df: pd.DataFrame, required_fields: List[str]) -> pd.DataFrame:
    """
    Remove rows that contain NaN in any of the specified required fields.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        required_fields (List[str]): A list of column names that must not contain NaN values.

    Returns:
        pd.DataFrame: A cleaned DataFrame with rows containing NaNs in required fields removed.
    """
    return df.dropna(subset=required_fields)


def fill_missing_continuous(df: pd.DataFrame, continuous_features: List[str]) -> pd.DataFrame:
    """
    Fill missing values in continuous feature columns with their respective column means.

    Args:
        df (pd.DataFrame): The input DataFrame containing missing values.
        continuous_features (List[str]): A list of column names considered as continuous features.

    Returns:
        pd.DataFrame: The DataFrame with missing values in continuous features filled.
    """
    df[continuous_features] = df[continuous_features].fillna(df[continuous_features].mean())
    return df


def remove_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Remove specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_remove (List[str]): A list of column names to be removed.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns dropped.
    """
    return df.drop(columns=columns_to_remove)


def clean_primary_col(df: pd.DataFrame, primary_field: Union[str, List[str]]) -> pd.DataFrame:
    """
    Clean the DataFrame by removing rows with missing values in the primary field(s)
    and dropping duplicate entries based on the primary key(s).

    Args:
        df (pd.DataFrame): The input DataFrame.
        primary_field (Union[str, List[str]]): A column name or list of column names that serve as the primary key(s).

    Returns:
        pd.DataFrame: A cleaned DataFrame with no missing or duplicate primary key values.
    """
    if isinstance(primary_field, str):
        primary_field = [primary_field]

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


def split_X_y(df: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into feature matrix X and target(s) y.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_cols (List[str]): List of target column(s) to extract.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features (X) and target(s) (y).
    """
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    return X, y


def prepare_train_val_test_sets(
    df: pd.DataFrame,
    split_cfg: Dict[str, float],
    target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the input DataFrame into training, validation, and test sets,
    and separate each into feature matrices (X) and target labels (y).

    Args:
        df (pd.DataFrame): The full DataFrame to split.
        split_cfg (Dict[str, float]): Dictionary with keys 'train', 'val', and 'test' specifying the split ratios.
        target_column (str): Target column.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame)
            - y_train (pd.DataFrame)
            - X_val (pd.DataFrame)
            - y_val (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_test (pd.DataFrame)
    """
    train_df, val_df, test_df = split_dataset(df, split_cfg, stratify_col=target_column)
    X_train, y_train = split_X_y(train_df, target_column)
    X_val, y_val = split_X_y(val_df, target_column)
    X_test, y_test = split_X_y(test_df, target_column)
    return X_train, y_train, X_val, y_val, X_test, y_test

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

    # Split Dataset
    X_train, y_train, *_ = prepare_train_val_test_sets(df, config["split"], features["target"])

    # Perform Random Forest
    rf_params = config["random_forest"]
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train.squeeze())
    joblib.dump(model, config["output"]["model_path"])
    

if __name__ == "__main__":
    main()
