import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from political_party_analysis.loader import DataLoader
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def mock_df() -> pd.DataFrame:
    data = {
        "id": [0, 1, 2, 2],
        "col1": [1, 2, 3, 3],
        "col2": [30.0, np.nan, 30.0, 30.0],
        "col3": [5, 7, 12, 12],
        "non_feature": ["a", "b", "c", "c"],
        "all_nans": [np.nan] * 4,
    }
    df = pd.DataFrame(data=data)
    return df

@pytest.fixture
def data_loader(mocker, mock_df: pd.DataFrame) -> DataLoader:
    mocker.patch.object(DataLoader, "_download_data", return_value = mock_df)
    return DataLoader()


#---unit test cases---

#data initialization
def test_data_loader_initialization(data_loader: DataLoader, mock_df: pd.DataFrame):
    """Test the initialization of mock data

    Args:
        data_loader (DataLoader): test instance of DataLoader
        mock_df (pd.DataFrame): mock data frame
    """
    assert isinstance(data_loader, DataLoader)
    assert data_loader.party_data.shape == mock_df.shape

#test remove duplicates
def test_remove_duplicates(data_loader: DataLoader, mock_df: pd.DataFrame):
    """Test the removal of duplicate rows from the data frame

    Args:
        data_loader (DataLoader): test instance of DataLoader
        mock_df (pd.DataFrame): mock data frame
    """
    
    deduped_df = data_loader.remove_duplicates(mock_df)
    assert deduped_df.shape == (3,6) #3 rows and 6 columns
    assert deduped_df.index.tolist() == [0,1,2]

#Test remove non-feature columns
def test_remove_nonfeature_cols(data_loader: DataLoader, mock_df: pd.DataFrame):
    """Test to remove non-feature columns

    Args:
        data_loader (DataLoader): test instance of DataLoader
        mock_df (pd.DataFrame): mock data frame
    """
    processed_df = data_loader.remove_nonfeature_cols(mock_df, ["non_feature","all_nans"], ["id"])
    assert processed_df.shape == (4, 3)
    assert processed_df.index.name == "id"
    assert "non_feature" not in processed_df.columns
    assert "all_nans" not in processed_df.columns


#Test handling NaN values
def test_handle_NaN_values(data_loader: DataLoader, mock_df: pd.DataFrame):
    # Call the method, which modifies the DataFrame in-place.
    # We do not use the return value to avoid potential NoneType errors.
    data_loader.handle_NaN_values(mock_df)

    # Assertions to verify the result on the original (now modified) DataFrame.
    assert mock_df.shape == (4, 6)
    assert mock_df.isnull().sum().sum() == 0




# data_loader = DataLoader()
# # Define the mock data needed for this specific block.
# mock_df = pd.DataFrame({
#     "id": [0, 1, 2, 2],
#     "col1": [1, 2, 3, 3],
#     "col2": [30.0, np.nan, 30.0, 30.0],
#     "col3": [5, 7, 12, 12],
#     "non_feature": ["a", "b", "c", "c"],
#     "all_nans": [np.nan] * 4,
# })

# test function for data scaling 
def test_scale_features(data_loader: DataLoader, mock_df: pd.DataFrame):
    numeric_df = mock_df[["col1","col2"]].drop_duplicates().fillna(0).reset_index(drop=True)
    scaled_df = data_loader.scale_features(numeric_df)
    scaler = StandardScaler()
    expected_scaled_array = scaler.fit_transform(numeric_df)
    expected_df = pd.DataFrame(expected_scaled_array, index=numeric_df.index, columns=numeric_df.columns)
    assert np.allclose(scaled_df.mean(), 0)
    assert np.allclose(scaled_df.std(ddof=0), 1)
    assert_frame_equal(scaled_df, expected_df)

@pytest.mark.integration
def test_download_data():
    data_loader = DataLoader()
    assert data_loader.party_data.shape == (277, 55)


def test_preprocess_data(mocker, mock_df: pd.DataFrame):
    data_loader = DataLoader()
    mocker.patch.object(data_loader, "party_data", mock_df)
    mocker.patch.object(data_loader, "non_features", ["non_feature", "all_nans"])
    mocker.patch.object(data_loader, "index", ["id"])
    data_loader.preprocess_data()
    expected_df = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [0.0] * 3,
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    expected_df.index.name = "id"
    assert_frame_equal(data_loader.party_data, expected_df, rtol=3)
