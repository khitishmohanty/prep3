import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    def __init__(self, method: str, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.method = method
        self.numeric_data = data.select_dtypes(include='number')
        self.non_numeric_data = data.select_dtypes(exclude='number')
    
    # transformation
    def transform(self) -> pd.DataFrame:
        
        #1. standardize the data 
        scalar = StandardScaler()
        scaled_numeric_data = scalar.fit_transform(self.numeric_data)
        
        #2. apply the dimentionality reduction
        pca = PCA(n_components=self.n_components)
        reducer_numeric_data = pca.fit_transform(scaled_numeric_data)
        self.model = pca
        
        #3. refactor the df
        pc_columns = [f'PC{i+1}' for i in range(self.n_components)]
        transformed_df = pd.DataFrame(reducer_numeric_data, columns=pc_columns, index=self.numeric_data.index)
        
        return transformed_df
        
