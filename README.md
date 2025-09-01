# Transformer Synthetic Data Generator (TGUI)

A comprehensive GUI application for generating high-quality synthetic tabular data using Transformer neural networks. This tool provides an intuitive interface for training transformer models and generating privacy-preserving synthetic datasets with customizable constraints and class balancing.

## Features

### Core Functionality
- **Transformer-based synthetic data generation** - Leverages attention mechanisms for high-quality tabular data synthesis
- **Interactive GUI** - User-friendly Tkinter-based interface for all operations
- **Automated hyperparameter tuning** - Grid search optimization with progress tracking
- **Advanced constraint system** - Set value ranges for numerical columns and allowed values for categorical columns
- **Class-specific generation** - Generate samples for specific target classes or balanced datasets
- **Dark/Light theme support** - Toggle between visual themes for comfortable usage

### Data Processing
- **Categorical and numerical data support** - Automatic handling of mixed data types
- **Flexible preprocessing** - MinMax scaling for numerical features and intelligent categorical encoding
- **Custom sequence length** - Adjustable sequence length for different dataset characteristics
- **Robust error handling** - Comprehensive validation and error recovery

### Model Features
- **Positional encoding** - Advanced positional encoding with dynamic expansion
- **Multi-head attention** - Configurable attention heads for complex pattern learning
- **Configurable architecture** - Adjustable model depth, dimensions, and feedforward layers
- **GPU acceleration** - Automatic CUDA detection and utilization when available

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch (with CUDA support for GPU acceleration)
- Required Python packages (see requirements below)

### Required Dependencies
```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib tkinter
pip install tqdm
```

### Setup
1. Clone or download all project files to a directory
2. Ensure all Python files are in the same directory:
   - `tgui.py` (main application)
   - `transformer_model.py` (core transformer implementation)
   - `tuning.py` (hyperparameter optimization)
   - `TGUI_*.py` (GUI modules)
   - `visualizations.py` (plotting utilities)

## Usage

### Starting the Application
Run the main application:
```bash
python tgui.py
```

### Basic Workflow

#### 1. Setup Paths
- Click **"Setup Paths"** to configure your data source and output location
- Select your CSV data file (semicolon-separated)
- Specify categorical columns (comma-separated list)
- Choose a save directory for results

#### 2. Generate Samples
- Click **"Generate Samples"** to open the generation interface
- Configure model parameters:
  - **Training settings**: epochs, batch size, learning rate
  - **Architecture**: model dimensions, attention heads, layers
  - **Generation settings**: sample count, noise factor, sampling method
- Set value constraints (optional):
  - **Numerical constraints**: min/max ranges
  - **Categorical constraints**: allowed values
- Choose to train a new model or use existing one
- Click **"Generate Samples"** to start the process

#### 3. Find Optimal Parameters
- Click **"Find Parameters"** to launch hyperparameter optimization
- Configure parameter ranges for grid search:
  - Model dimensions, attention heads, layer counts
  - Learning rates, batch sizes, dropout rates
- Set the number of training epochs for each combination
- Monitor progress and stop early if needed

### Advanced Features

#### Value Constraints
Set realistic bounds on generated data:
- **Numerical columns**: Define min/max ranges (e.g., age: 18-80)
- **Categorical columns**: Specify allowed values (e.g., gender: 'M,F')

#### Class-Specific Generation
Generate samples for specific target classes:
- Specify the target class column name
- Enter the desired class value
- The model will generate samples predominantly of that class

#### Categorical Sampling Methods
Choose how categorical values are selected:
- **Nearest**: Deterministic selection of closest valid category
- **Probabilistic**: Random selection weighted by proximity to predicted value

## File Structure

```
project/
│
├── tgui.py                          # Main application entry point
├── transformer_model.py             # Core transformer model implementation
├── tuning.py                        # Hyperparameter optimization
├── visualizations.py                # Training plot generation
│
├── TGUI_constraints_window.py       # Constraints configuration GUI
├── TGUI_find_parameters.py         # Parameter search GUI
├── TGUI_generate_samples.py        # Sample generation GUI
├── TGUI_setup_paths.py             # Path configuration GUI
├── TGUI_transformer_backend.py     # Backend interface
├── TGUI_globals.py                 # Global variables (not shown)
├── TGUI_theme.py                   # Theme management (not shown)
└── TGUI_tooltip.py                 # Tooltip utilities (not shown)
```

## Model Architecture

The application uses a custom Transformer architecture optimized for tabular data:

- **Positional Encoding**: Custom implementation with dynamic sequence length adaptation
- **Multi-Head Attention**: Configurable attention heads for pattern recognition
- **Encoder-Only Architecture**: Simplified transformer design for sequence modeling
- **Embedding Layer**: Linear projection from input features to model dimensions
- **Output Layer**: Linear projection back to original feature space

### Default Configuration
```python
{
    'd_model': 128,           # Model embedding dimension
    'nhead': 4,              # Number of attention heads
    'num_encoder_layers': 2,  # Number of transformer layers
    'dim_feedforward': 256,   # Feedforward network dimension
    'dropout': 0.1,          # Dropout rate
    'max_seq_length': 50,    # Maximum sequence length
    'batch_size': 32,        # Training batch size
    'learning_rate': 0.001,  # Adam optimizer learning rate
    'epochs': 50             # Training epochs
}
```

## Data Format

### Input Requirements
- **Format**: CSV files with semicolon (`;`) separation
- **Structure**: Each row represents a data sample, each column a feature
- **Mixed types**: Support for both numerical and categorical columns
- **Missing values**: Automatic handling with median imputation for numerical features

### Example CSV Structure
```csv
age;height;weight;gender;income_bracket;education
25;170.5;65.2;M;2;Bachelor
32;162.1;58.7;F;3;Master
...
```

## Output

### Generated Files
- **Synthetic samples**: `Generated_Transformer_Samples_{count}.csv`
- **Trained models**: `tabular_transformer_model.pth`
- **Tuning results**: `tuning_results.csv`
- **Training plots**: `loss_curve_{combination}.png`, `val_r2_{combination}.png`

### Sample Output Structure
Generated CSV files maintain the same column structure as input data with realistic synthetic values that preserve statistical relationships while protecting privacy.

## Hyperparameter Tuning

The application includes comprehensive hyperparameter optimization:

### Tuning Process
1. **Grid Search**: Systematic exploration of parameter combinations
2. **Cross-Validation**: Training/validation split for robust evaluation
3. **Progress Tracking**: Real-time progress updates and early stopping
4. **Result Visualization**: Automatic generation of training curves
5. **Best Model Selection**: Automatic identification of optimal parameters

### Tunable Parameters
- Model architecture (dimensions, heads, layers)
- Training hyperparameters (learning rate, batch size, dropout)
- Sequence processing (max sequence length)

## Technical Details

### Model Training
- **Loss Function**: Mean Squared Error (MSE) for sequence prediction
- **Optimizer**: Adam with configurable learning rate
- **Validation**: R² score and validation loss tracking
- **Early Stopping**: Manual termination support during long training runs

### Data Processing Pipeline
1. **Loading**: CSV parsing with error handling
2. **Categorical Encoding**: Automatic mapping of categorical values to indices
3. **Normalization**: MinMax scaling for numerical features
4. **Sequence Creation**: Sliding window approach for temporal modeling
5. **Inverse Transform**: Reconstruction of original scale and categories

### Generation Process
1. **Seed Selection**: Random or specific sequence initialization
2. **Autoregressive Generation**: Step-by-step sample creation
3. **Constraint Application**: Real-time constraint enforcement
4. **Post-processing**: Scale restoration and categorical mapping

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)
- **Storage**: Sufficient space for datasets and generated samples
- **GPU**: CUDA-compatible GPU recommended for faster training (optional)

## Known Limitations

- **Sequence Length**: Very long sequences may require significant memory
- **Large Datasets**: Memory usage scales with dataset size
- **Complex Relationships**: Some intricate data relationships may not be perfectly captured
- **Categorical Constraints**: Limited to exact value matching for categorical features

## Troubleshooting

### Common Issues
- **Memory Errors**: Reduce batch size or sequence length
- **Training Instability**: Lower learning rate or increase dropout
- **Poor Quality**: Increase training epochs or model complexity
- **Constraint Violations**: Check constraint format and value ranges

### Debug Mode
Enable detailed logging by modifying the logging level in `tgui.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This project welcomes contributions for:
- Additional constraint types
- New sampling methods
- Performance optimizations
- Extended model architectures
- Enhanced visualization features

## License

Please refer to the project repository for licensing information.

## Acknowledgments

Built using PyTorch for deep learning capabilities and Tkinter for the graphical user interface. The transformer architecture is adapted for tabular data synthesis with custom positional encoding and constraint handling.