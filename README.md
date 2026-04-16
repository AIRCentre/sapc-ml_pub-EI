# Deep Learning Prediction of *Pithomyces chartarum* Sporulation

[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2FXXXXX-blue)](https://doi.org/10.XXXX/XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning framework for predicting *Pithomyces chartarum* sporulation risk in Terceira Island, Azores (Portugal), using time series analysis of meteorological and topographic data.

## Overview

*Pithomyces chartarum* is a fungus that produces sporidesmin, a toxin causing pithomycotoxicosis (facial eczema) in grazing livestock. This disease poses significant threats to animal health and agricultural economies. This repository contains the implementation of a convolutional neural network (CNN) model that achieved 0.87 AUC in predicting sporulation risk, outperforming traditional threshold-based approaches.

### Key Features

- **Deep Learning Approach**: Automated time series classification using mcfly framework
- **Multiple Architectures**: Tests CNN, DeepConvLSTM, ResNet, and InceptionTime models
- **Long-term Environmental History**: Uses 365 days of preceding meteorological conditions
- **High Performance**: Best model achieves 0.87 AUC on independent test data
- **Feature Importance**: Identifies mean daily temperature as the primary driver

## Research Context

This work is part of the Atlantic International Research Centre's efforts to enhance early warning systems for livestock protection in the Azores. The model is designed to integrate with existing operational alert systems currently available at [https://aircentre.io/app/sap/](https://aircentre.io/app/sap/).

### Study Area

- **Location**: Terceira Island, Azores Archipelago (Portugal)
- **Climate**: Temperate oceanic with high humidity (annual average 80%)
- **Data Period**: 2023 (1,149 georeferenced spore count samples)
- **Risk Classification**: Low Risk (≤10,000 spores/gram) vs Early Alert (>10,000 spores/gram)

## Installation

### Requirements

- Python 3.9 or higher
- conda (recommended) or pip

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/AIRCentre/sapc-ml_pub-EI.git
cd sapc-ml_pub-EI

# Create and activate conda environment
conda create -n sapc-ml python=3.9 -y
conda activate sapc-ml

# Install scientific computing packages
conda install -c conda-forge numpy pandas scipy scikit-learn tensorflow -y

# Install additional packages
pip install mcfly pyarrow
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/AIRCentre/sapc-ml_pub-EI.git
cd sapc-ml_pub-EI

# Create virtual environment
python -m venv sapc-env
source sapc-env/bin/activate  # On Windows: sapc-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Files

We provide both conda and pip environment specifications:

- **environment.yml**: Conda environment file
- **requirements.txt**: Pip requirements file

Create the conda environment directly from file:
```bash
conda env create -f environment.yml
conda activate sapc-ml
```

## Usage

### Basic Execution

```bash
# Ensure your environment is activated
conda activate sapc-ml  # or: source sapc-env/bin/activate

# Run the model
python main_.py
```

### Expected Runtime

The script performs the following steps:

1. **Data Loading**: Loads spore counts, meteorological, and topographic data (~1 minute)
2. **Model Selection**: Tests 20 different neural network architectures (30-60 minutes)
3. **Final Training**: Trains best model on full dataset (~5-10 minutes)
4. **Prediction & Export**: Generates predictions and saves model (~1 minute)

**Total runtime**: 45-90 minutes depending on hardware

### Output Files

The script generates several files in the `results/` directory:

- `modelcomparisons.csv`: Performance comparison of all 20 candidate models
- `PREDICTIONS.csv`: Final model predictions on test set
- `SAP.json`: Model architecture in JSON format
- `sap_model_complete.keras`: Complete trained model (weights + architecture)

## Data Structure

```
sapc-ml_pub-EI/
├── main_.py                    # Main analysis script
├── data/                       # Input data directory
│   ├── agrid.csv              # Terceira Island grid
│   ├── sporulation.feather    # Spore count data
│   ├── meteo_to_iuri.feather  # Meteorological data
│   ├── terceira_elevation.csv # Elevation data
│   ├── slope_values.csv       # Slope data
│   ├── aspect_values.csv      # Aspect data
│   └── terceira/              # Shapefiles (optional)
├── results/                    # Output directory
├── environment.yml            # Conda environment
├── requirements.txt           # Pip requirements
└── README.md                  # This file
```

## Methodology

### Data Sources

- **Spore Counts**: 1,149 samples collected by UNICOL across Terceira Island (2023)
- **Meteorology**: 59 IoT weather stations measuring temperature and humidity
- **Topography**: Digital elevation model derived slope and aspect data

### Model Architecture

The framework uses **mcfly** for automated deep learning model selection:

- **Input**: 365-day time series of meteorological + topographic variables
- **Architectures Tested**: CNN, DeepConvLSTM, ResNet, InceptionTime (20 models total)
- **Training Strategy**: 50% train / 25% validation / 25% test split
- **Performance Metric**: Area Under the ROC Curve (AUC)

### Best Model Performance

- **Architecture**: Convolutional Neural Network (CNN)
- **Validation AUC**: 0.815
- **Test AUC**: 0.87
- **Key Finding**: Mean daily temperature is the most important predictor

## Results

### Model Comparison

| Architecture    | Train AUC | Val AUC | Val Loss |
|----------------|-----------|---------|----------|
| **CNN (Best)** | **0.843** | **0.815** | **3.501** |
| DeepConvLSTM   | 0.872     | 0.691   | 12.247   |
| CNN            | 0.815     | 0.806   | 0.904    |
| InceptionTime  | 0.584     | 0.659   | 10.174   |

### Feature Importance

1. **Mean Daily Temperature**: AUC reduction of 0.31 when randomised
2. **Relative Humidity**: AUC reduction of 0.03 when randomised  
3. **Topographic Variables**: Negligible impact (elevation, slope, aspect)

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{diogo2026deep,
  title={A deep learning approach to predicting Pithomyces chartarum sporulation for early disease warning},
  author={Diogo, Iúri and Capinha, César and Pinelo, João and Domingues, Elizabeth and Ávila, Mariana},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI when available]}
}
```

## Contributing

This research was conducted as part of a scientific study. While the code is made available for reproducibility, we welcome:

- Bug reports via GitHub Issues
- Questions about methodology or implementation
- Suggestions for improvements or extensions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

This research was funded by the *Agenda Mobilizadora* New Space Portugal, as part of Portugal's Recovery and Resilience Plan (RRP) - Project nº 02/C05-i01.01/2022.PC644936537-00000046; IAPMEI Projeto Nº11.

## Acknowledgments

- **TERINOV**: Access to IoT weather station data
- **Municipality of Angra do Heroísmo**: LoRaWAN network development support
- **Regional Veterinary Laboratory**: Consistent spore count analysis
- **UNICOL**: Field sample collection and domain expertise

## Authors

- **Iúri Diogo** - *Atlantic International Research Centre* - [iuri.diogo@aircentre.org](mailto:iuri.diogo@aircentre.org)
- **César Capinha** - *Centre of Geographical Studies, University of Lisbon*
- **João Pinelo** - *Atlantic International Research Centre*
- **Elizabeth Domingues** - *UNICOL-Cooperativa Agrícola*
- **Mariana Ávila** - *Atlantic International Research Centre*

## Related Links

- [AIR Centre](https://www.aircentre.org/)
- [Operational Alert System](https://aircentre.io/app/sap/)
- [mcfly Documentation](https://mcfly.readthedocs.io/)
