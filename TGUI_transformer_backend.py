import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Tuple, Any, Optional
import TGUI_globals as globals
from transformer_model import SyntheticDataGenerator


class TransformerBackend:
    """Backend handler for transformer-based synthetic data generation"""
    
    def __init__(self):
        self.generator = None
        self.scaled_data = None
        self.dataset = None
        self.model_path = None
        self.column_names = []
        self.categorical_columns = []
    
    def initialize_generator(self, 
                           data_path: str,
                           categorical_columns: List[str],
                           model_config: Dict[str, Any]) -> Tuple[List[str], bool]:
        """
        Initialize the synthetic data generator
        
        Returns:
            Tuple of (column_names, success_flag)
        """
        try:
            self.generator = SyntheticDataGenerator(model_config=model_config)
            
            # Load and preprocess data
            self.dataset, self.scaled_data = self.generator.load_and_preprocess_data(
                csv_path=data_path,
                seq_length=model_config.get('max_seq_length', 30),
                categorical_columns=categorical_columns
            )
            
            self.column_names = self.generator.column_names or []
            self.categorical_columns = self.generator.categorical_columns or []
            
            return self.column_names, True
            
        except Exception as e:
            print(f"Error initializing generator: {e}")
            return [], False
    
    def train_model(self, 
                   epochs: int,
                   batch_size: int,
                   save_path: str) -> bool:
        """
        Train the transformer model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save the trained model
            
        Returns:
            Success flag
        """
        try:
            if self.generator is None or self.dataset is None:
                raise ValueError("Generator not initialized")
            
            # Create data loader
            train_loader = DataLoader(
                self.dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True
            )
            
            if len(train_loader) == 0:
                raise ValueError("DataLoader is empty - not enough data or batch size too large")
            
            # Train the model
            history = self.generator.train(train_loader, epochs=epochs)
            
            # Save the model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.generator.save_model(save_path)
            self.model_path = save_path
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def load_existing_model(self, model_path: str) -> bool:
        """
        Load an existing trained model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Success flag
        """
        try:
            if self.generator is None:
                raise ValueError("Generator not initialized")
            
            self.generator.load_model(model_path)
            self.model_path = model_path
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_samples(self,
                        num_samples: int,
                        target_class: Optional[str] = None,
                        target_class_column: Optional[str] = None,
                        value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None,
                        generation_noise_factor: float = 0.01,
                        categorical_sampling_method: str = 'nearest',
                        analyze_ranges: bool = True) -> pd.DataFrame:
        """
        Generate synthetic samples
        
        Args:
            num_samples: Number of samples to generate
            target_class: Specific class value to generate (optional)
            target_class_column: Name of the class column (required if target_class is specified)
            value_constraints: Dictionary of constraints for columns
            generation_noise_factor: Noise factor for generation
            categorical_sampling_method: Method for categorical sampling ('nearest' or 'probabilistic')
            analyze_ranges: Whether to analyze and print ranges
            
        Returns:
            DataFrame with generated samples
        """
        try:
            if self.generator is None:
                raise ValueError("Generator not initialized")
            
            if self.generator.model is None:
                raise ValueError("Model not trained or loaded")
            
            if self.scaled_data is None or len(self.scaled_data) < self.generator.seq_length:
                raise ValueError("Not enough data for generation")
            
            # Get random seed data
            max_start_idx = len(self.scaled_data) - self.generator.seq_length
            seed_start_idx = np.random.randint(0, max_start_idx)
            seed_data = torch.FloatTensor(
                self.scaled_data[seed_start_idx:seed_start_idx + self.generator.seq_length]
            )
            
            # Generate samples based on whether target class is specified
            if target_class is not None and target_class_column is not None:
                # Generate with specific class
                synthetic_df = self.generator.generate_synthetic_data_with_class(
                    seed_data=seed_data,
                    num_samples=num_samples,
                    class_column=target_class_column,
                    class_value=target_class,
                    value_constraints=value_constraints,
                    generation_noise_factor=generation_noise_factor,
                    categorical_sampling_method=categorical_sampling_method,
                    analyze_ranges=analyze_ranges
                )
            else:
                # Generate mixed samples
                synthetic_df = self.generator.generate_synthetic_data(
                    seed_data=seed_data,
                    num_samples=num_samples,
                    value_constraints=value_constraints,
                    generation_noise_factor=generation_noise_factor,
                    categorical_sampling_method=categorical_sampling_method,
                    analyze_ranges=analyze_ranges
                )
            
            # Return only the generated part (excluding seed)
            if len(synthetic_df) > self.generator.seq_length:
                return synthetic_df.iloc[self.generator.seq_length:].reset_index(drop=True)
            else:
                return synthetic_df.reset_index(drop=True)
            
        except Exception as e:
            print(f"Error generating samples: {e}")
            raise
    
    def generate_balanced_samples(self,
                                 class_column: str,
                                 samples_per_class: Dict[str, int],
                                 value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None,
                                 generation_noise_factor: float = 0.01,
                                 categorical_sampling_method: str = 'nearest',
                                 analyze_ranges: bool = True) -> pd.DataFrame:
        """
        Generate balanced samples across multiple classes
        
        Args:
            class_column: Name of the class column
            samples_per_class: Dictionary mapping class values to number of samples
            value_constraints: Dictionary of constraints for columns
            generation_noise_factor: Noise factor for generation
            categorical_sampling_method: Method for categorical sampling
            analyze_ranges: Whether to analyze and print ranges
            
        Returns:
            DataFrame with balanced generated samples
        """
        try:
            if self.generator is None:
                raise ValueError("Generator not initialized")
            
            if self.generator.model is None:
                raise ValueError("Model not trained or loaded")
            
            if self.scaled_data is None or len(self.scaled_data) < self.generator.seq_length:
                raise ValueError("Not enough data for generation")
            
            # Get random seed data
            max_start_idx = len(self.scaled_data) - self.generator.seq_length
            seed_start_idx = np.random.randint(0, max_start_idx)
            seed_data = torch.FloatTensor(
                self.scaled_data[seed_start_idx:seed_start_idx + self.generator.seq_length]
            )
            
            # Generate balanced samples
            balanced_df = self.generator.generate_synthetic_data_balancing_classes(
                seed_data=seed_data,
                class_column=class_column,
                samples_per_class=samples_per_class,
                value_constraints=value_constraints,
                generation_noise_factor=generation_noise_factor,
                categorical_sampling_method=categorical_sampling_method,
                analyze_ranges_per_class=analyze_ranges
            )
            
            return balanced_df
            
        except Exception as e:
            print(f"Error generating balanced samples: {e}")
            raise
    
    def get_available_class_values(self, class_column: str) -> List[str]:
        """
        Get available values for a categorical class column
        
        Args:
            class_column: Name of the class column
            
        Returns:
            List of available class values
        """
        try:
            if self.generator is None:
                return []
            
            if class_column in self.generator.categorical_mappings:
                return list(self.generator.categorical_mappings[class_column].values())
            else:
                return []
                
        except Exception as e:
            print(f"Error getting class values: {e}")
            return []
    
    def save_samples(self, samples_df: pd.DataFrame, output_path: str) -> bool:
        """
        Save generated samples to CSV
        
        Args:
            samples_df: DataFrame with generated samples
            output_path: Path to save the CSV file
            
        Returns:
            Success flag
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            samples_df.to_csv(output_path, sep=';', index=False)
            return True
            
        except Exception as e:
            print(f"Error saving samples: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.generator is None:
            return {}
        
        return {
            'input_dim': self.generator.input_dim,
            'seq_length': self.generator.seq_length,
            'column_names': self.column_names,
            'categorical_columns': self.categorical_columns,
            'config': self.generator.config,
            'model_loaded': self.generator.model is not None
        }
