"""
This module provides functions to load, reformat, and save Raman spectra data.
It includes the following functionalities:

1. Loading and reformatting spectra data from a .txt file.
2. Adding a timestamp column to the DataFrame.
3. Saving the DataFrame to a .csv file.

"""

import pandas as pd
import numpy as np


def load_and_reformat_spectra(file_path: str) -> pd.DataFrame:
    """
    Load and reformat spectra data from a .txt file.
    Parameters:
    file_path (str): The path to the .txt file containing the spectra data.
    Returns:
    pd.DataFrame: The reformatted DataFrame.
    """
    # Load the data from the .txt file
    data = np.loadtxt(file_path, skiprows=0)

    # Extract the first column as headers

    headers = data[:, 0]  # Raman Shift in cm-1
    spectra = data[:, 1:]  # Intensity

    # Transpose the rest of the DataFrame
    spectra_transposed = spectra.T

    # Convert the transposed array to a DataFrame and assign the headers
    df_transposed = pd.DataFrame(spectra_transposed, columns=headers)

    return df_transposed


def add_timestamp_column(df: pd.DataFrame, interval: float) -> pd.DataFrame:
    """
    Add a timestamp column to the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame to which the timestamp column will be
    added.
    interval (int): The interval in seconds between each timestamp.
    Returns:
    pd.DataFrame: The DataFrame with the added timestamp column.
    """
    num_rows = df.shape[0]
    timestamps = np.arange(0, num_rows * interval, interval)
    df.insert(0, 'time', timestamps)
    return df


def save_dataframe_to_csv(df: pd.DataFrame, csv_file_path: str) -> None:
    """
    Save the DataFrame to a .csv file.
    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    csv_file_path (str): The path to save the .csv file.
    """
    df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully saved to {csv_file_path}")


def main() -> None:
    """
    Main function to execute the data loading, reformatting,
    timestamp addition, and saving to CSV.

    This function performs the following steps:
    1. Defines the file paths for input and output.
    2. Specifies the time interval between each timestamp.
    3. Loads and reformats the spectra data from a .txt file.
    4. Adds a timestamp column to the DataFrame.
    5. Saves the reformatted DataFrame to a .csv file.
    """

    # Define the file paths
    input_file_path = './data/raw/Raman_spectra_data.txt'
    output_csv_file_path = (
        './data/processed/Raman_spectra_data_reformatted.csv'
    )

    interval = 0.04562  # Time interval in seconds as defined in the paper
    # Load and reformat the spectra data
    df = load_and_reformat_spectra(input_file_path)
    # Add the timestamp column
    df_with_timestamp = add_timestamp_column(df, interval)
    # Save the reformatted DataFrame to a .csv file
    save_dataframe_to_csv(df_with_timestamp, output_csv_file_path)


if __name__ == "__main__":
    main()
