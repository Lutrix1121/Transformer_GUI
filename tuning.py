import os
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import DataLoader, Subset
from transformer_model import SyntheticDataGenerator

def tune_hyperparameters(csv_path: str, param_grid: dict, seq_length: int = None, 
                        tuning_epochs: int = 10, val_split: float = 0.2, 
                        categorical_columns: list = None, results_csv: str = "tuning_results.csv",
                        progress_callback=None, stop_flag=None):
    """
    Automates hyperparameter tuning with progress indication and improved stopping mechanism.
    
    Args:
        csv_path: Path to the CSV file
        param_grid: dictionary where keys are hyperparameter names and values are lists of possible values.
        seq_length: sequence length for the model (if None, will be determined automatically)
        tuning_epochs: number of epochs to run for each combination during tuning.
        val_split: proportion of the dataset to use for validation.
        categorical_columns: list of column names to treat as categorical.
        results_csv: path where to save the results CSV
        progress_callback: optional callback function for progress updates (current, total)
        stop_flag: callable that returns True when search should be stopped
        
    Returns:
        tuple: (best_params, best_history)
    """
    print(f"Starting hyperparameter tuning...")
    print(f"CSV path: {csv_path}")
    print(f"Results will be saved to: {results_csv}")
    
    # Check stop flag at the beginning
    if stop_flag and stop_flag():
        print("Tuning process stopped before starting")
        return None, None
    
    # Initialize generator to load data
    generator_dummy = SyntheticDataGenerator()
    dataset, original_data = generator_dummy.load_and_preprocess_data(
        csv_path, seq_length, categorical_columns
    )
    
    # Check stop flag after data loading
    if stop_flag and stop_flag():
        print("Tuning process stopped after data loading")
        return None, None
    
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    indices = list(range(total_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    results = []
    best_params = None
    best_val_loss = float('inf')
    best_history = None
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_combinations = list(product(*values))
    total_combinations = len(all_combinations)
    
    print(f"Total parameter combinations to test: {total_combinations}")
    
    for idx, vals in enumerate(all_combinations, start=1):
        # Check stop flag at the beginning of each combination
        if stop_flag and stop_flag():
            print("Tuning process stopped by user")
            break
        
        params = dict(zip(keys, vals))
        print(f"\nTuning progress: {idx}/{total_combinations} - Testing combination: {params}")
        
        # Update progress if callback provided
        if progress_callback:
            try:
                progress_callback(idx-1, total_combinations)
            except:
                pass  # Continue even if progress callback fails
        
        # Check stop flag again before starting training
        if stop_flag and stop_flag():
            print("Tuning process stopped during parameter combination")
            break
        
        # Create configuration for this combination
        config = {
            'd_model': params.get('d_model', generator_dummy.config['d_model']),
            'nhead': params.get('nhead', generator_dummy.config['nhead']),
            'num_encoder_layers': params.get('num_encoder_layers', generator_dummy.config['num_encoder_layers']),
            'dim_feedforward': params.get('dim_feedforward', generator_dummy.config['dim_feedforward']),
            'dropout': params.get('dropout', generator_dummy.config['dropout']),
            'batch_size': params.get('batch_size', generator_dummy.config['batch_size']),
            'learning_rate': params.get('learning_rate', generator_dummy.config['learning_rate']),
            'epochs': tuning_epochs,
            'max_seq_length': params.get('max_seq_length', generator_dummy.config['max_seq_length'])
        }
        
        try:
            # Create new generator with this configuration
            generator = SyntheticDataGenerator(config)
            
            # Pass categorical info along
            generator.categorical_columns = categorical_columns if categorical_columns else []
            generator.categorical_mappings = generator_dummy.categorical_mappings
            generator.input_dim = generator_dummy.input_dim       
            generator.column_names = generator_dummy.column_names
            
            # Create data loaders
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
            
            # Build model
            generator.build_model()
            
            # Check stop flag before training
            if stop_flag and stop_flag():
                print("Tuning process stopped before training model")
                break
            
            # Train model with stop flag checking
            history = generator.train(train_loader, val_loader, epochs=tuning_epochs, stop_flag=stop_flag)
            
            # Check if training was stopped
            if stop_flag and stop_flag():
                print("Tuning process stopped during model training")
                break
            
            # Get final validation metrics
            final_val_loss = history['val_loss'][-1] if history and history['val_loss'] else float('inf')
            final_val_r2 = history['val_r2'][-1] if history and history['val_r2'] else None
            
            print(f"Combination {params} final validation loss: {final_val_loss:.6f}")
            
            # Store results
            result_row = {
                **params,
                'final_val_loss': final_val_loss,
                'final_val_r2': final_val_r2,
                'status': 'completed'
            }
            results.append(result_row)
            
            # Update best parameters if this is better
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_params = params
                best_history = history
                print(f"New best parameters found! Loss: {best_val_loss:.6f}")
            
        except Exception as e:
            # Check if the exception is due to stopping
            if stop_flag and stop_flag():
                print("Tuning process stopped during training")
                break
                
            print(f"Error with combination {params}: {str(e)}")
            # Store failed result
            result_row = {
                **params,
                'final_val_loss': float('inf'),
                'final_val_r2': None,
                'status': f'failed: {str(e)}'
            }
            results.append(result_row)
        
        # Save results after every combination (if not stopped)
        if not (stop_flag and stop_flag()):
            try:
                df_results = pd.DataFrame(results)
                df_results.to_csv(results_csv, index=False)
                print(f"Results updated in {results_csv}")
            except Exception as e:
                print(f"Warning: Could not save results to CSV: {e}")
    
    # Check if process was stopped
    if stop_flag and stop_flag():
        print("Hyperparameter tuning was stopped by user")
        # Still save partial results
        if results:
            try:
                df_results = pd.DataFrame(results)
                df_results.to_csv(results_csv, index=False)
                print(f"Partial results saved to {results_csv}")
            except Exception as e:
                print(f"Warning: Could not save partial results to CSV: {e}")
        return best_params, best_history
    
    # Final progress update
    if progress_callback:
        try:
            progress_callback(total_combinations, total_combinations)
        except:
            pass
    
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*50)
    
    if best_params:
        print("Best Hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
    else:
        print("No successful parameter combinations found!")
    
    print(f"Detailed results saved to: {results_csv}")
    
    return best_params, best_history