from pathlib import Path
from typing import List
from urllib.request import urlretrieve
import logging
from sklearn.preprocessing import StandardScaler

import pandas as pd



class DataLoader:
    """Class to load the political parties dataset"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        logging.info("Removing duplicate values")
        df.drop_duplicates(keep='first', inplace=True)
        return df

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        logging.info("Removing non-feature colums")
        df = df.drop(columns=non_features)
        logging.info("setting up index")
        df.set_index(index, inplace=True)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        df.fillna(0, inplace=True)
        return df
  

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        scaler = StandardScaler()
        scalar_data = scaler.fit_transform(df)
        return pd.DataFrame(scalar_data, index=df.index, columns = df.columns)
     

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        df = self.party_data
        df = self.remove_duplicates(df)
        df = self.remove_nonfeature_cols(df, self.non_features, self.index)
        df = self.handle_NaN_values(df)
        df = self.scale_features(df)
        self.party_data = df
       
