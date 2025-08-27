import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from transformer_model import SyntheticDataGenerator
from visualizations import save_training_plots


class HyperparameterTuner:
    """
    A class for automated hyperparameter tuning with GUI compatibility.
    """
    
    def __init__(self):
        """Initialize the hyperparameter tuner."""
        self.results = []
        self.best_params = None
        self.best_val_loss = float('inf')
        self.best_history = None
        self.current_combination = 0
        self.total_combinations = 0
        self.is_running = False
        self.stop_requested = False
        
        # Data-related attributes
        self.dataset = None
        self.original_data = None
        self.generator_dummy = None
        
    def load_data(self, csv_path: str, seq_length: int = None, categorical_columns: list = None):
        """
        Load and preprocess the data.
        
        Args:
            csv_path: Path to the CSV file
            seq_length: Sequence length for the model (if None, will be determined automatically)
            categorical_columns: List of column names to treat as categorical
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize generator to load data
            self.generator_dummy = SyntheticDataGenerator()
            self.dataset, self.original_data = self.generator_dummy.load_and_preprocess_data(
                csv_path, seq_length, categorical_columns
            )
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_param_grid(self, param_grid: dict):
        """
        Setup the parameter grid and calculate total combinations.
        
        Args:
            param_grid: Dictionary where keys are hyperparameter names and values are lists of possible values
        """
        self.param_grid = param_grid
        keys, values = zip(*param_grid.items())
        all_combinations = list(product(*values))
        self.total_combinations = len(all_combinations)
        self.all_combinations = [(dict(zip(keys, vals)), idx) for idx, vals in enumerate(all_combinations)]
        
    def create_data_loaders(self, val_split: float = 0.2):
        """
        Create training and validation data loaders.
        
        Args:
            val_split: Proportion of the dataset to use for validation
            
        Returns:
            tuple: (train_indices, val_indices, train_size, val_size)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        total_samples = len(self.dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        indices = list(range(total_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        return train_indices, val_indices, train_size, val_size
    
    def test_single_combination(self, params: dict, tuning_epochs: int, train_indices: list, val_indices: list):
        """
        Test a single parameter combination.
        
        Args:
            params: Dictionary of parameters for this combination
            tuning_epochs: Number of epochs to run for training
            train_indices: Indices for training data
            val_indices: Indices for validation data
            
        Returns:
            dict: Result dictionary with metrics and status
        """
        # Create configuration for this combination
        config = {
            'd_model': params.get('d_model', self.generator_dummy.config['d_model']),
            'nhead': params.get('nhead', self.generator_dummy.config['nhead']),
            'num_encoder_layers': params.get('num_encoder_layers', self.generator_dummy.config['num_encoder_layers']),
            'dim_feedforward': params.get('dim_feedforward', self.generator_dummy.config['dim_feedforward']),
            'dropout': params.get('dropout', self.generator_dummy.config['dropout']),
            'batch_size': params.get('batch_size', self.generator_dummy.config['batch_size']),
            'learning_rate': params.get('learning_rate', self.generator_dummy.config['learning_rate']),
            'epochs': tuning_epochs,
            'max_seq_length': params.get('max_seq_length', self.generator_dummy.config['max_seq_length'])
        }
        
        try:
            # Create new generator with this configuration
            generator = SyntheticDataGenerator(config)
            
            # Pass categorical info along
            generator.categorical_columns = self.generator_dummy.categorical_columns
            generator.categorical_mappings = self.generator_dummy.categorical_mappings
            generator.reverse_categorical_mappings = self.generator_dummy.reverse_categorical_mappings
            generator.scalers = self.generator_dummy.scalers
            generator.input_dim = self.generator_dummy.input_dim       
            generator.column_names = self.generator_dummy.column_names
            generator.seq_length = self.generator_dummy.seq_length
            
            # Create data loaders
            train_subset = Subset(self.dataset, train_indices)
            val_subset = Subset(self.dataset, val_indices)

            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
            
            if len(train_loader) == 0 or len(val_loader) == 0:
                return {
                    **params,
                    'final_train_loss': float('inf'),
                    'final_val_loss': float('inf'),
                    'final_val_r2': None,
                    'status': 'insufficient_data'
                }
            
            # Build model
            generator.build_model()
            
            # Train model with stop flag checking
            history = generator.train(train_loader, val_loader, epochs=tuning_epochs, 
                                    stop_flag=lambda: self.stop_requested)
            
            # Check if training was stopped
            if self.stop_requested:
                return {
                    **params,
                    'final_train_loss': float('inf'),
                    'final_val_loss': float('inf'),
                    'final_val_r2': None,
                    'status': 'stopped'
                }
            
            # Get final validation metrics
            final_train_loss = history['train_loss'][-1] if history and history['train_loss'] else float('inf')
            final_val_loss = history['val_loss'][-1] if history and history['val_loss'] else float('inf')
            final_val_r2 = history['val_r2'][-1] if history and history['val_r2'] else None
            
            if final_val_loss is None:
                final_val_loss = float('inf')
            
            # Update best parameters if this is better
            if final_val_loss != float('inf') and final_val_loss < self.best_val_loss:
                self.best_val_loss = final_val_loss
                self.best_params = params.copy()
                self.best_history = history
            
            results_dir = os.path.join(os.path.dirname("tuning_results"), "tuning_results", "plots")
            save_training_plots(history, results_dir, combo_idx=len(self.results) + 1)

            return {
                **params,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_val_r2': final_val_r2,
                'status': 'completed'
            }
            
        except Exception as e:
            return {
                **params,
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_val_r2': None,
                'status': f'failed: {str(e)}'
            }
    
    def run_tuning(self, csv_path: str, param_grid: dict, seq_length: int = None, 
                   tuning_epochs: int = 10, val_split: float = 0.2, 
                   categorical_columns: list = None, results_csv: str = "tuning_results.csv",
                   progress_callback=None, stop_flag=None):
        """
        Run the complete hyperparameter tuning process.
        
        Args:
            csv_path: Path to the CSV file
            param_grid: Dictionary where keys are hyperparameter names and values are lists of possible values
            seq_length: Sequence length for the model (if None, will be determined automatically)
            tuning_epochs: Number of epochs to run for each combination during tuning
            val_split: Proportion of the dataset to use for validation
            categorical_columns: List of column names to treat as categorical
            results_csv: Path where to save the results CSV
            progress_callback: Optional callback function for progress updates (current, total)
            stop_flag: Callable that returns True when search should be stopped
            
        Returns:
            tuple: (best_params, best_history)
        """
        print(f"Starting hyperparameter tuning...")
        print(f"CSV path: {csv_path}")
        print(f"Results will be saved to: {results_csv}")
        
        # Reset state
        self.is_running = True
        self.stop_requested = False
        self.results = []
        self.best_params = None
        self.best_val_loss = float('inf')
        self.best_history = None
        self.current_combination = 0
        
        try:
            # Load data
            if not self.load_data(csv_path, seq_length, categorical_columns):
                return None, None
            
            # Setup parameter grid
            self.setup_param_grid(param_grid)
            
            # Create data loaders
            train_indices, val_indices, train_size, val_size = self.create_data_loaders(val_split)
            print(f"Training samples: {train_size}, Validation samples: {val_size}")
            print(f"Total parameter combinations to test: {self.total_combinations}")
            
            # Test each combination
            for params, idx in self.all_combinations:
                # Check stop conditions
                if stop_flag and stop_flag():
                    self.stop_requested = True
                    break
                if self.stop_requested:
                    break
                
                self.current_combination = idx + 1
                print(f"\nTuning progress: {self.current_combination}/{self.total_combinations} - Testing combination: {params}")
                
                # Update progress if callback provided
                if progress_callback:
                    try:
                        progress_callback(idx, self.total_combinations)
                    except:
                        pass  # Continue even if progress callback fails
                
                # Test this combination
                result = self.test_single_combination(params, tuning_epochs, train_indices, val_indices)
                self.results.append(result)
                
                if result['status'] == 'completed':
                    print(f"Combination {params} final validation loss: {result['final_val_loss']:.6f}")
                    if result['final_val_loss'] < self.best_val_loss:
                        print(f"New best parameters found! Loss: {result['final_val_loss']:.6f}")
                elif result['status'] == 'stopped':
                    break
                else:
                    print(f"Combination failed: {result['status']}")
                
                # Save results after every combination (if not stopped)
                if not self.stop_requested:
                    self.save_results(results_csv)
            
            # Final progress update
            if progress_callback and not self.stop_requested:
                try:
                    progress_callback(self.total_combinations, self.total_combinations)
                except:
                    pass
            
            # Print final results
            self.print_final_results(results_csv)
            
        except Exception as e:
            print(f"Error during tuning: {e}")
        finally:
            self.is_running = False
        
        return self.best_params, self.best_history
    
    def save_results(self, results_csv: str):
        """
        Save current results to CSV file.
        
        Args:
            results_csv: Path to save the results CSV
        """
        try:
            if self.results:
                df_results = pd.DataFrame(self.results)
                df_results.to_csv(results_csv, index=False)
                print(f"Results updated in {results_csv}")
        except Exception as e:
            print(f"Warning: Could not save results to CSV: {e}")
    
    def print_final_results(self, results_csv: str):
        """
        Print final results summary.
        
        Args:
            results_csv: Path where results were saved
        """
        print("\n" + "="*50)
        if self.stop_requested:
            print("HYPERPARAMETER TUNING STOPPED BY USER")
        else:
            print("HYPERPARAMETER TUNING COMPLETED")
        print("="*50)
        
        if self.best_params:
            print("Best Hyperparameters:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")
            print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        else:
            print("No successful parameter combinations found!")
        
        print(f"Detailed results saved to: {results_csv}")
    
    def stop(self):
        """Request the tuning process to stop."""
        self.stop_requested = True
    
    def get_progress(self):
        """
        Get current progress information.
        
        Returns:
            tuple: (current_combination, total_combinations, is_running)
        """
        return self.current_combination, self.total_combinations, self.is_running
    
    def get_best_results(self):
        """
        Get the best results found so far.
        
        Returns:
            tuple: (best_params, best_val_loss, best_history)
        """
        return self.best_params, self.best_val_loss, self.best_history
    
    def get_all_results(self):
        """
        Get all results collected so far.
        
        Returns:
            list: List of result dictionaries
        """
        return self.results.copy()


# Legacy function for backward compatibility
def tune_hyperparameters(csv_path: str, param_grid: dict, seq_length: int = None, 
                        tuning_epochs: int = 10, val_split: float = 0.2, 
                        categorical_columns: list = None, results_csv: str = "tuning_results.csv",
                        progress_callback=None, stop_flag=None):
    """
    Legacy function wrapper for the HyperparameterTuner class.
    This maintains backward compatibility with existing code.
    """
    tuner = HyperparameterTuner()
    return tuner.run_tuning(
        csv_path=csv_path,
        param_grid=param_grid,
        seq_length=seq_length,
        tuning_epochs=tuning_epochs,
        val_split=val_split,
        categorical_columns=categorical_columns,
        results_csv=results_csv,
        progress_callback=progress_callback,
        stop_flag=stop_flag
    )