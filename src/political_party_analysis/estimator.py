import pandas as pd
from sklearn.neighbors import KernelDensity


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names

    #1. Write a function to model the distribution of the political party dataset
    def model_distribution(self, kernel='gaussian', bandwidth = 0.5):
        self.kde_model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kde_model.fit(self.data)
    
    
    #2. Write a function to randomly sample 10 parties from this distribution
    def sample_from_distribution(self,n_sample=10) -> pd.DataFrame:
        samples = self.kde_model.sample(n_sample)
        new_sample_df = pd.DataFrame(samples, columns=self.data.columns)
        return new_sample_df


    #3. Map the randomly sampled 10 parties back to the original higher dimensional
    def map_to_high_dimention_space(self, low_dim_sample: pd.DataFrame) -> pd.DataFrame:
        high_dim_sample_np = self.dim_reducer_model.inverse_transform(low_dim_sample)
        high_dim_sample_df = pd.DataFrame(high_dim_sample_np, column = self.feature_names)
        return high_dim_sample_df