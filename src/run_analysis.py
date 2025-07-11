from pathlib import Path

from matplotlib import pyplot
import logging

from political_party_analysis.loader import DataLoader
from political_party_analysis.visualization import scatter_plot, plot_finnish_parties
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator

if __name__ == "__main__":
    
    plots_dir = Path(__file__).parents[1].joinpath("plots")
    plots_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logging.info("Loading data")
    data_loader = DataLoader()
    # Data pre-processing step
    
    original_data = data_loader.party_data
    logging.info(f"Original data shape loaded as: {data_loader.party_data.shape}")
    
    data_loader.preprocess_data()
    processed_data = data_loader.party_data
    logging.info(f"Data processed with the shape as: {data_loader.party_data.shape}")
    
    

    # Dimensionality reduction step
    dim_reducer = DimensionalityReducer(
        data = processed_data,
        method = "PCA",
        n_components= 2
    )
    
    reduced_dim_data = dim_reducer.transform()

    ## Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    high_dim_features = processed_data.columns
    estimator = DensityEstimator(
        data= processed_data, dim_reducer=dim_reducer, high_dim_feature_names=high_dim_features
    )
    estimator.model_distribution(kernel='gaussian', bandwidth = 0.5)
    new_samples = estimator.sample_from_distribution(n_sample=100)

    # Plot density estimation results here
    ##### YOUR CODE GOES HERE #####
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    pyplot.figure(figsize=(12,10))
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    scatter_plot(
        new_samples,
        color="b",
        splot=splot,
        label="dim reduced data",
    )
  
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    plot_data = reduced_dim_data.reset_index()
    finnish_data = plot_data[plot_data['country'] == 'fin']
    # scatter_plot(
    #     finnish_data,
    #     color="b",
    #     splot=splot,
    #     label="Finnish parties",
    # )
    
    scatter_plot(finnish_data)
    #plot_finnish_parties(plot_data)

    print("Analysis Complete")
