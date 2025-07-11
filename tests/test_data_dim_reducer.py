import pandas as pd
import pytest
import logging
from political_party_analysis.dim_reducer import DimensionalityReducer

@pytest.fixture
def mock_df() -> pd.DataFrame:
    df = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [-1.175, -0.1, 1.257],
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    df.index.name = "id"
    return df

@pytest.fixture
def mock_df_with_non_numeric() -> pd.DataFrame:
    df = pd.DataFrame(
        data={
            "party_name": ["A", "B", "C"],
            "score1": [1.22,0,2.3],
            "score2": [1.222,1,-2.3],
            "country": ["X", "Y", "Z"]
        },
        index=[1,2,3],
    )
    df.index.name = "id"
    return df
    
def test_initialization(mock_df: pd.DataFrame):
    dim_reducer = DimensionalityReducer("PCA", mock_df)
    assert dim_reducer.data.equals(mock_df)
    assert dim_reducer.n_components == 2


def test_dimensionality_reducer(mock_df: pd.DataFrame):
    dim_reducer = DimensionalityReducer("PCA", mock_df)
    transformed_data = dim_reducer.transform()
    assert transformed_data.shape == (mock_df.shape[0], 2)
    
    
def test_handle_non_numeric_data(mock_df_with_non_numeric: pd.DataFrame):
    dim_reducer = DimensionalityReducer("PCA", mock_df_with_non_numeric)
    transform_data = dim_reducer.transform()
    
    expected_num_cols = 2
    assert transform_data.shape == (mock_df_with_non_numeric.shape[0] ,expected_num_cols)
    assert "PC1" in transform_data.columns
    assert "PC2" in transform_data.columns