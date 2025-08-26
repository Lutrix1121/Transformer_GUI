import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from typing import List, Dict, Optional, Union, Tuple, Any
import os # Added for path checking

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 100):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def _create_positional_encoding(self, max_seq_length: int, d_model: int):
        """Create or recreate the positional encoding buffer"""
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=True)
    
    def forward(self, x):
        seq_len = x.size(1)
        # If the input sequence is longer than our current PE buffer, recreate it
        if seq_len > self.pe.size(1):
            print(f"Expanding positional encoding from {self.pe.size(1)} to {seq_len}")
            self._create_positional_encoding(seq_len, self.d_model)
            self.max_seq_length = seq_len
        # Use only the needed portion of the positional encoding
        return x + self.pe[:, :seq_len]

class TabularTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1, max_seq_length: int = 100):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, input_dim)
        
    def forward(self, src, src_mask=None):
        x = self.embedding(src) 
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_mask)
        output = self.output_layer(x)
        return output

class TabularDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int): self.data = data; self.seq_length = seq_length
    def __len__(self): return max(0, len(self.data) - self.seq_length)
    def __getitem__(self, idx):
        if idx + self.seq_length + 1 > len(self.data): idx = len(self.data) - self.seq_length - 1
        if idx < 0: raise IndexError("Dataset too small.")
        x = self.data[idx : idx + self.seq_length]; y = self.data[idx + 1 : idx + self.seq_length + 1]; return torch.FloatTensor(x), torch.FloatTensor(y)


class SyntheticDataGenerator:
    def __init__(self, model_config: Dict = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.config = {
            'd_model': 128, 'nhead': 4, 'num_encoder_layers': 2,
            'dim_feedforward': 256, 'dropout': 0.1, 'max_seq_length': 50,
            'batch_size': 32, 'learning_rate': 0.001, 'epochs': 50,
            'generation_noise_factor': 0.01, # Default noise level
            'categorical_sampling_method': 'nearest' # 'nearest' or 'probabilistic'
        }
        if model_config: self.config.update(model_config)
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.categorical_mappings: Dict[str, Dict[int, Any]] = {}
        self.reverse_categorical_mappings: Dict[str, Dict[Any, int]] = {}
        self.column_names: Optional[List[str]] = None
        self.input_dim: Optional[int] = None
        self.model: Optional[TabularTransformer] = None
        self.categorical_columns: List[str] = []
        self.seq_length: int = self.config['max_seq_length']
        self.processed_constraints: Optional[Dict[int, Dict]] = None


    def load_and_preprocess_data(self, csv_path: str, seq_length: Optional[int] = None, categorical_columns: Optional[List[str]] = None):
        try: df = pd.read_csv(csv_path, sep=';')
        except Exception as e: print(f"Error reading CSV: {e}"); raise
        self.column_names = df.columns.tolist()
        self.categorical_columns = [col for col in categorical_columns if col in df.columns] if categorical_columns else []
        scaled_data = np.zeros_like(df.values, dtype=np.float32)
        self.categorical_mappings, self.reverse_categorical_mappings, self.scalers = {}, {}, {}
        for i, col in enumerate(self.column_names):
            if col in self.categorical_columns:
                unique_values = df[col].astype(str).unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}; inverse_mapping = {idx: val for val, idx in mapping.items()}
                self.categorical_mappings[col] = inverse_mapping; self.reverse_categorical_mappings[col] = mapping
                col_data = df[col].astype(str).map(mapping).values.astype(np.float32)
                num_unique = len(unique_values); scaled_col_data = col_data / (num_unique - 1) if num_unique > 1 else np.zeros_like(col_data)
                scaled_data[:, i] = scaled_col_data
            else:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.isnull().any(): median_val = numeric_col.median(); numeric_col = numeric_col.fillna(median_val)
                if numeric_col.isnull().any(): raise ValueError(f"Unresolvable NaNs in column {col}")
                scaler = MinMaxScaler(); col_data_reshaped = numeric_col.values.reshape(-1, 1)
                scaled_data[:, i] = scaler.fit_transform(col_data_reshaped).flatten(); self.scalers[col] = scaler
        self.input_dim = scaled_data.shape[1]
        self.seq_length = seq_length if seq_length is not None else min(self.config['max_seq_length'], max(1, len(scaled_data) // 10))
        self.config['max_seq_length'] = self.seq_length; print(f"Using sequence length: {self.seq_length}")
        dataset = TabularDataset(scaled_data, self.seq_length); return dataset, scaled_data

    def build_model(self):
        if self.input_dim is None: raise ValueError("input_dim not set.")
        self.model = TabularTransformer(input_dim=self.input_dim, d_model=self.config['d_model'], nhead=self.config['nhead'], num_encoder_layers=self.config['num_encoder_layers'], dim_feedforward=self.config['dim_feedforward'], dropout=self.config['dropout'], max_seq_length=self.config['max_seq_length']).to(self.device); return self.model

    def train(self, train_loader, val_loader=None, epochs: Optional[int] = None, stop_flag = None):
        if self.model is None: self.build_model()
        epochs = epochs if epochs is not None else self.config['epochs']
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            if stop_flag and stop_flag():
                print(f"Training stopped by user at epoch {epoch+1}")
                break

            # Training phase
            self.model.train()
            epoch_loss = 0
            processed_batches = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                # Check stop flag every 10 batches to avoid excessive checking
                if batch_idx % 10 == 0 and stop_flag and stop_flag():
                    print(f"Training stopped by user at epoch {epoch+1}, batch {batch_idx}")
                    return history
                    
                if x.ndim != 3 or y.ndim != 3 or x.shape[1] != self.seq_length or y.shape[1] != self.seq_length: 
                    continue
                    
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                
                if output.shape != y.shape: 
                    continue
                    
                loss = criterion(output, y)
                if torch.isnan(loss): 
                    continue
                    
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                processed_batches += 1
                
            if processed_batches == 0: 
                print(f"Epoch {epoch+1}/{epochs} - No batches processed.")
                # Still append values to maintain history consistency
                history['train_loss'].append(float('inf'))
                history['val_loss'].append(float('inf'))
                history['val_r2'].append(None)
                continue
                
            train_loss = epoch_loss / processed_batches
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_info = ""
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_processed = 0
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for x, y in val_loader:
                        if x.ndim != 3 or y.ndim != 3 or x.shape[1] != self.seq_length or y.shape[1] != self.seq_length:
                            continue
                            
                        x, y = x.to(self.device), y.to(self.device)
                        output = self.model(x)
                        
                        if output.shape != y.shape:
                            continue
                            
                        loss = criterion(output, y)
                        if torch.isnan(loss):
                            continue
                            
                        val_loss += loss.item()
                        val_processed += 1
                        
                        # Collect predictions and targets for R2 calculation
                        all_predictions.extend(output.cpu().numpy().flatten())
                        all_targets.extend(y.cpu().numpy().flatten())
                
                if val_processed > 0:
                    avg_val_loss = val_loss / val_processed
                    history['val_loss'].append(avg_val_loss)
                    
                    # Calculate R2 score
                    if len(all_predictions) > 0 and len(all_targets) > 0:
                        try:
                            val_r2 = r2_score(all_targets, all_predictions)
                            history['val_r2'].append(val_r2)
                            val_info = f" - Val Loss: {avg_val_loss:.6f} - Val R2: {val_r2:.4f}"
                        except:
                            history['val_r2'].append(None)
                            val_info = f" - Val Loss: {avg_val_loss:.6f} - Val R2: N/A"
                    else:
                        history['val_r2'].append(None)
                        val_info = f" - Val Loss: {avg_val_loss:.6f} - Val R2: N/A"
                else:
                    history['val_loss'].append(float('inf'))
                    history['val_r2'].append(None)
                    val_info = " - Val: No valid batches"
            else:
                # No validation loader provided
                history['val_loss'].append(None)
                history['val_r2'].append(None)
                
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}{val_info}')
            
        print("Training finished.")
        return history

    def _inverse_transform_generated(self, generated_data: np.ndarray) -> pd.DataFrame:
        original_scale_data = np.empty_like(generated_data, dtype=object);
        if self.column_names is None: raise ValueError("Column names not set.")
        for i, col in enumerate(self.column_names):
            col_data = generated_data[:, i]
            if col in self.categorical_columns:
                if col not in self.categorical_mappings: original_scale_data[:, i] = col_data; continue
                inverse_mapping = self.categorical_mappings[col]; num_unique = len(inverse_mapping)
                denormalized_vals = col_data * (num_unique - 1) if num_unique > 1 else np.zeros_like(col_data)
                indices = np.clip(np.rint(denormalized_vals).astype(int), 0, num_unique - 1 if num_unique > 0 else 0)
                original_col = np.array([inverse_mapping.get(idx, f"Unknown_Idx_{idx}") for idx in indices]); original_scale_data[:, i] = original_col
            else:
                if col not in self.scalers: original_scale_data[:, i] = col_data; continue
                scaler = self.scalers[col]; original_vals = scaler.inverse_transform(col_data.reshape(-1, 1)).flatten(); original_scale_data[:, i] = np.round(original_vals, 2)
        return pd.DataFrame(original_scale_data, columns=self.column_names)

    def analyze_generated_data_ranges(self, synthetic_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if self.column_names is None: raise ValueError("Column names not available.")
        ranges = {}
        for col in self.column_names:
            if col not in synthetic_df.columns: continue
            if col in self.categorical_columns:
                unique_values = sorted(synthetic_df[col].astype(str).unique().tolist()); ranges[col] = {'type': 'categorical', 'values': unique_values}
            else:
                numeric_col = pd.to_numeric(synthetic_df[col], errors='coerce')
                if numeric_col.isnull().all(): min_val, max_val, col_type = "N/A", "N/A", 'non-numeric'
                elif numeric_col.notna().any(): min_val, max_val, col_type = numeric_col.min(), numeric_col.max(), 'numerical'
                else: min_val, max_val, col_type = np.nan, np.nan, 'numerical (all NaN)'
                ranges[col] = {'type': col_type, 'min': min_val, 'max': max_val}
        return ranges

    def print_ranges(self, ranges: Dict[str, Dict[str, Any]]):
        for col, info in ranges.items():
            if info['type'] == 'numerical':
                min_str = f"{info['min']:.2f}" if isinstance(info['min'], (int, float)) and not np.isnan(info['min']) else str(info['min'])
                max_str = f"{info['max']:.2f}" if isinstance(info['max'], (int, float)) and not np.isnan(info['max']) else str(info['max'])
                print(f"- {col} (Numerical): Min={min_str}, Max={max_str}")
            elif info['type'] == 'categorical':
                values_str = ", ".join(map(str, info['values'][:10]));
                if len(info['values']) > 10: values_str += f", ... ({len(info['values'])} total)"
                print(f"- {col} (Categorical): Values=[{values_str}]")
            else: print(f"- {col} ({info['type']}): Min={info['min']}, Max={info['max']}")

    def _get_normalized_class_value(self, class_column: str, class_value: Any) -> float:
        if class_column not in self.categorical_columns: raise ValueError(f"'{class_column}' not categorical.")
        if class_column not in self.reverse_categorical_mappings: raise ValueError(f"Mapping for '{class_column}' not found.")
        mapping = self.reverse_categorical_mappings[class_column]; class_value_str = str(class_value)
        if class_value_str not in mapping: raise ValueError(f"Value '{class_value_str}' not found in mapping for '{class_column}'. Available: {list(mapping.keys())}")
        class_idx = mapping[class_value_str]; num_unique = len(mapping); return float(class_idx) / (num_unique - 1) if num_unique > 1 else 0.0

    def _find_column_index(self, column_name: str) -> int:
        if self.column_names is None: raise ValueError("Column names not set.")
        try: return self.column_names.index(column_name)
        except ValueError: raise ValueError(f"Column '{column_name}' not found.")

    def _preprocess_constraints(self, value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None) -> Optional[Dict[int, Dict]]:
        if value_constraints is None: return None
        if self.column_names is None: raise ValueError("Cannot process constraints before loading data.")
        processed = {}
        for col_name, constraint in value_constraints.items():
            try: col_idx = self._find_column_index(col_name)
            except ValueError: print(f"Warning: Constraint for unknown column '{col_name}'. Ignoring."); continue
            if col_name in self.categorical_columns:
                if not isinstance(constraint, list): print(f"Warning: Cat constraint for '{col_name}' not list. Ignoring."); continue
                if col_name not in self.reverse_categorical_mappings: print(f"Warning: No mapping for cat '{col_name}'. Ignoring constraint."); continue
                allowed_norm_values = []; mapping = self.reverse_categorical_mappings[col_name]; num_unique = len(mapping)
                if num_unique <= 1: continue
                for val in constraint:
                    val_str = str(val)
                    if val_str not in mapping: print(f"Warning: Allowed value '{val_str}' for '{col_name}' not found. Ignoring."); continue
                    norm_val = float(mapping[val_str]) / (num_unique - 1); allowed_norm_values.append(norm_val)
                if not allowed_norm_values: print(f"Warning: No valid allowed values for '{col_name}'. Constraint ignored."); continue
                processed[col_idx] = {'type': 'categorical', 'allowed_norm': sorted(list(set(allowed_norm_values)))}
            else:
                if not isinstance(constraint, (tuple, list)) or len(constraint) != 2: print(f"Warning: Num constraint for '{col_name}' not tuple/list(2). Ignoring."); continue
                min_val, max_val = constraint
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)): print(f"Warning: Min/max for '{col_name}' not numbers. Ignoring."); continue
                if min_val >= max_val: print(f"Warning: Min >= max for '{col_name}'. Ignoring."); continue
                if col_name not in self.scalers: print(f"Warning: No scaler for '{col_name}'. Ignoring constraint."); continue
                scaler = self.scalers[col_name]; norm_min = scaler.transform(np.array([[min_val]]))[0, 0]; norm_max = scaler.transform(np.array([[max_val]]))[0, 0]
                norm_min = np.clip(norm_min, 0.0, 1.0); norm_max = np.clip(norm_max, 0.0, 1.0)
                processed[col_idx] = {'type': 'numerical', 'norm_min': norm_min, 'norm_max': norm_max}
        return processed if processed else None

    def _apply_constraints_to_step(self, next_step: torch.Tensor,
                                   processed_constraints: Optional[Dict[int, Dict]],
                                   categorical_sampling_method: str = 'nearest', # 'nearest' or 'probabilistic'
                                   forced_class_idx: Optional[int] = None,
                                   forced_class_norm_val: Optional[float] = None,
                                   epsilon: float = 1e-6) -> torch.Tensor: # Epsilon for probabilistic sampling
        """Applies preprocessed constraints to a single predicted step (inplace)."""
        if processed_constraints is None and forced_class_idx is None:
            return next_step

        constrained_step = next_step.clone()

        for col_idx in range(constrained_step.shape[0]):
            # --- Priority: Forced class value ---
            if col_idx == forced_class_idx and forced_class_norm_val is not None:
                constrained_step[col_idx] = forced_class_norm_val
                continue # Skip other constraints for this forced column

            # --- General Constraints ---
            if processed_constraints and col_idx in processed_constraints:
                constraint = processed_constraints[col_idx]
                current_val = constrained_step[col_idx].item()

                if constraint['type'] == 'numerical':
                    norm_min = constraint['norm_min']
                    norm_max = constraint['norm_max']
                    constrained_val = np.clip(current_val, norm_min, norm_max)
                    constrained_step[col_idx] = constrained_val

                elif constraint['type'] == 'categorical':
                    allowed_norm = constraint['allowed_norm']
                    if not allowed_norm: continue

                    if len(allowed_norm) == 1:
                        # If only one category is allowed, snap to it
                        snapped_val = allowed_norm[0]
                    elif categorical_sampling_method == 'probabilistic':
                        # Probabilistic sampling based on distance
                        distances = np.array([abs(current_val - allowed) for allowed in allowed_norm])
                        # Convert distance to closeness score (invert, add epsilon)
                        scores = 1.0 / (distances + epsilon)
                        # Normalize scores to probabilities
                        probabilities = scores / np.sum(scores)
                        # Handle potential NaN probabilities if all scores are zero (shouldn't happen with epsilon)
                        if np.isnan(probabilities).any():
                             probabilities = np.ones_like(allowed_norm) / len(allowed_norm) # Equal probability fallback

                        # Sample based on probabilities
                        chosen_idx = np.random.choice(len(allowed_norm), p=probabilities)
                        snapped_val = allowed_norm[chosen_idx]
                    else: # Default to 'nearest'
                        # Find the closest allowed normalized value
                        distances = np.array([abs(current_val - allowed) for allowed in allowed_norm])
                        closest_idx = np.argmin(distances)
                        snapped_val = allowed_norm[closest_idx]

                    constrained_step[col_idx] = snapped_val

        # Final safety clamp
        constrained_step = torch.clamp(constrained_step, 0.0, 1.0)
        return constrained_step



    def generate_synthetic_data(self, seed_data: torch.Tensor, num_samples: int,
                                value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None,
                                generation_noise_factor: Optional[float] = None, # Use instance default if None
                                categorical_sampling_method: Optional[str] = None, # Use instance default if None
                                analyze_ranges: bool = False) -> pd.DataFrame:
        """Generates data with constraints, noise control, and categorical sampling options."""
        if self.model is None: raise ValueError("Model not trained/loaded.")
        # Use instance defaults if parameters are not provided
        noise_factor = generation_noise_factor if generation_noise_factor is not None else self.config.get('generation_noise_factor', 0.01)
        cat_sampling = categorical_sampling_method if categorical_sampling_method is not None else self.config.get('categorical_sampling_method', 'nearest')

        processed_constraints = self._preprocess_constraints(value_constraints)
        if processed_constraints: print(f"Applying constraints (Noise: {noise_factor}, CatSampling: {cat_sampling}).")

        self.model.eval(); current_sequence = seed_data.clone().to(self.device); generated_normalized = []
        with torch.no_grad():
            for _ in range(num_samples):
                input_seq = current_sequence.unsqueeze(0); predictions = self.model(input_seq)
                next_step_raw = predictions[0, -1, :]
                # Add noise (controlled by factor)
                noise = torch.randn_like(next_step_raw) * noise_factor
                next_step_noisy = next_step_raw + noise
                # Apply constraints (with categorical sampling method)
                next_step_constrained = self._apply_constraints_to_step(
                    next_step_noisy, processed_constraints, cat_sampling
                )
                generated_normalized.append(next_step_constrained.cpu().numpy())
                current_sequence = torch.cat([current_sequence[1:], next_step_constrained.unsqueeze(0)], dim=0)

        generated_normalized_array = np.array(generated_normalized)
        full_normalized_data = np.vstack([seed_data.cpu().numpy(), generated_normalized_array])
        synthetic_df = self._inverse_transform_generated(full_normalized_data)
        if analyze_ranges and num_samples > 0:
            generated_part_df = self._inverse_transform_generated(generated_normalized_array)
            print("\n--- Analysis of Generated Data Ranges (Constrained, excluding seed) ---")
            ranges = self.analyze_generated_data_ranges(generated_part_df); self.print_ranges(ranges)
            print("---------------------------------------------------------------------\n")
        return synthetic_df


    def generate_synthetic_data_with_class(self, seed_data: torch.Tensor, num_samples: int,
                                          class_column: str, class_value: Any,
                                          value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None,
                                          generation_noise_factor: Optional[float] = None,
                                          categorical_sampling_method: Optional[str] = None,
                                          analyze_ranges: bool = False) -> pd.DataFrame:
        """Generates data forcing a class, with constraints, noise control, and cat sampling."""
        if self.model is None: raise ValueError("Model not trained/loaded.")
        noise_factor = generation_noise_factor if generation_noise_factor is not None else self.config.get('generation_noise_factor', 0.01)
        cat_sampling = categorical_sampling_method if categorical_sampling_method is not None else self.config.get('categorical_sampling_method', 'nearest')

        processed_constraints = self._preprocess_constraints(value_constraints)
        if processed_constraints: print(f"Applying constraints (Noise: {noise_factor}, CatSampling: {cat_sampling}).")
        forced_col_idx = self._find_column_index(class_column)
        forced_norm_val = self._get_normalized_class_value(class_column, class_value)
        print(f"Forcing class '{class_column}' to '{class_value}' (norm: {forced_norm_val:.4f}).")

        self.model.eval(); current_sequence = seed_data.clone().to(self.device); generated_normalized = []
        with torch.no_grad():
            for _ in range(num_samples):
                input_seq = current_sequence.unsqueeze(0); predictions = self.model(input_seq)
                next_step_raw = predictions[0, -1, :]
                noise = torch.randn_like(next_step_raw) * noise_factor
                next_step_noisy = next_step_raw + noise
                # Apply constraints (passing forced class info and cat sampling method)
                next_step_constrained = self._apply_constraints_to_step(
                    next_step_noisy, processed_constraints, cat_sampling,
                    forced_class_idx=forced_col_idx, forced_class_norm_val=forced_norm_val
                )
                generated_normalized.append(next_step_constrained.cpu().numpy())
                current_sequence = torch.cat([current_sequence[1:], next_step_constrained.unsqueeze(0)], dim=0)

        generated_normalized_array = np.array(generated_normalized)
        full_normalized_data = np.vstack([seed_data.cpu().numpy(), generated_normalized_array])
        synthetic_df = self._inverse_transform_generated(full_normalized_data)
        synthetic_df.loc[self.seq_length:, class_column] = str(class_value) # Final enforcement
        if analyze_ranges and num_samples > 0:
            generated_part_df = self._inverse_transform_generated(generated_normalized_array)
            temp_analysis_df = generated_part_df.copy(); temp_analysis_df[class_column] = str(class_value)
            print(f"\n--- Analysis (Forced Class '{class_value}', Constrained, excluding seed) ---")
            ranges = self.analyze_generated_data_ranges(temp_analysis_df); self.print_ranges(ranges)
            print("--------------------------------------------------------------------------\n")
        return synthetic_df


    def generate_synthetic_data_balancing_classes(self, seed_data: torch.Tensor,
                                                 class_column: str,
                                                 samples_per_class: Dict[Any, int],
                                                 value_constraints: Optional[Dict[str, Union[Tuple[float, float], List[Any]]]] = None,
                                                 generation_noise_factor: Optional[float] = None,
                                                 categorical_sampling_method: Optional[str] = None,
                                                 analyze_ranges_per_class: bool = False) -> pd.DataFrame:
        """Generates balanced data with constraints, noise control, and cat sampling."""
        all_generated_dfs = []
        base_seed_data = seed_data.clone()
        # Use instance defaults if parameters are not provided
        noise_factor = generation_noise_factor if generation_noise_factor is not None else self.config.get('generation_noise_factor', 0.01)
        cat_sampling = categorical_sampling_method if categorical_sampling_method is not None else self.config.get('categorical_sampling_method', 'nearest')

        for class_value, num_samples in samples_per_class.items():
            if num_samples <= 0: continue
            print(f"\nGenerating {num_samples} samples for class '{class_value}' (Noise: {noise_factor}, CatSampling: {cat_sampling})...")
            class_full_df = self.generate_synthetic_data_with_class(
                seed_data=base_seed_data, num_samples=num_samples,
                class_column=class_column, class_value=class_value,
                value_constraints=value_constraints,
                generation_noise_factor=noise_factor, # Pass down specific values
                categorical_sampling_method=cat_sampling, # Pass down specific values
                analyze_ranges=analyze_ranges_per_class
            )
            generated_part_df = class_full_df.iloc[self.seq_length:].reset_index(drop=True)
            all_generated_dfs.append(generated_part_df)

        if not all_generated_dfs: return pd.DataFrame(columns=self.column_names)
        final_df = pd.concat(all_generated_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        print(f"\n--- Analysis of Final Balanced & Constrained Dataset Ranges ({len(final_df)} total samples) ---")
        final_ranges = self.analyze_generated_data_ranges(final_df); self.print_ranges(final_ranges)
        print("---------------------------------------------------------------------------------------\n")
        return final_df

    # --- save_model, load_model ---
    # (Unchanged from previous version - copy them)
    def save_model(self, path: str):
        if self.model is None: raise ValueError("No model to save.")
        model_state = {
            'model_state_dict': self.model.state_dict(), 'config': self.config, 'input_dim': self.input_dim,
            'column_names': self.column_names, 'scalers': self.scalers, 'categorical_mappings': self.categorical_mappings,
            'reverse_categorical_mappings': self.reverse_categorical_mappings, 'categorical_columns': self.categorical_columns,
            'seq_length': self.seq_length }
        try: torch.save(model_state, path); print(f"Model saved to {path}")
        except Exception as e: print(f"Error saving model: {e}")

    def load_model(self, path: str):
        try:
            model_state = torch.load(path, map_location=self.device)
            self.config = model_state['config'] # Load entire config, includes defaults for new params if saved model is older
            self.input_dim = model_state['input_dim']; self.column_names = model_state['column_names']
            self.scalers = model_state['scalers']; self.categorical_mappings = model_state.get('categorical_mappings', {})
            self.reverse_categorical_mappings = model_state.get('reverse_categorical_mappings')

            if not self.reverse_categorical_mappings and self.categorical_mappings:
                 self.reverse_categorical_mappings = { col: {v: k for k, v in inv_map.items()} for col, inv_map in self.categorical_mappings.items() }
            
            self.categorical_columns = model_state.get('categorical_columns', [])
            self.seq_length = model_state.get('seq_length', self.config.get('max_seq_length', 50))

            current_seq_length = self.config.get('max_seq_length', self.seq_length)
            self.seq_length = current_seq_length
            self.config['max_seq_length'] = current_seq_length
            self.config.setdefault('generation_noise_factor', 0.01)
            self.config.setdefault('categorical_sampling_method', 'nearest')

            try:
                self.model.load_state_dict(model_state['model_state_dict'])
                print(f"Model loaded from {path} (Seq Len: {self.seq_length})")
            except RuntimeError as e:
                if "size mismatch" in str(e) and "positional" in str(e).lower():
                    print(f"Positional encoding size mismatch detected. Rebuilding with current seq_length={self.seq_length}")
                    # Load state dict but skip the positional encoding buffer
                    state_dict = model_state['model_state_dict']
                    # Remove positional encoding buffer from saved state
                    state_dict = {k: v for k, v in state_dict.items() if 'positional_encoding.pe' not in k}
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"Model loaded from {path} with rebuilt positional encoding (Seq Len: {self.seq_length})")
                else:
                    raise e
            
            self.model.eval()

        except FileNotFoundError: print(f"Error: Model file not found at {path}"); raise
        except Exception as e: print(f"Error loading model: {e}"); raise


# --- Example Usage ---
    def start_generating(
        data_path = 'balanced_Dataset_cleaned.csv',
        model_savepath = 'tabular_transformer_model_v3.pth', # New save path
        contrained_synthetic_data_broad = 'constrained_synthetic_data_broad.csv',
        balanced_constrained_synthetic_data_broad = 'balanced_constrained_synthetic_data_broad.csv',
        categorical_cols = ["sex", "hear_left", "hear_right", "SMK_stat_type_cd", "DRK_YN"],
        target_class_column = 'DRK_YN',
        model_params = {
            'd_model': 64, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128,
            'dropout': 0.1, 'max_seq_length': 30, 'batch_size': 64,
            'learning_rate': 0.001, 'epochs': 50,
            # Set default generation params here if desired
            'generation_noise_factor': 0.1, # <<< INCREASED DEFAULT NOISE
            'categorical_sampling_method': 'probabilistic' # <<< USE PROBABILISTIC SAMPLING
        },
        train_new_model = True
    ):
        
        generator = SyntheticDataGenerator(model_config=model_params)

        print("Loading and preprocessing data...")
        try:
            dataset, scaled_data = generator.load_and_preprocess_data(
                data_path, seq_length=generator.config['max_seq_length'], categorical_columns=categorical_cols
            )
            train_loader = DataLoader(dataset, batch_size=generator.config['batch_size'], shuffle=True, drop_last=True)
        except Exception as e: print(f"Failed to load data: {e}"); exit()

        if train_new_model or not os.path.exists(model_savepath):
            print("\nTraining the model...")
            if len(train_loader) > 0:
                try:
                    history = generator.train(train_loader, epochs=generator.config['epochs'])
                    print("\nSaving the model..."); generator.save_model(model_savepath)
                except Exception as e: print(f"Error during training/saving: {e}"); exit()
            else: print("Error: DataLoader is empty."); exit()
        else:
            print("\nLoading existing model...");
            try: generator.load_model(model_savepath)
            except Exception as e: print(f"Failed to load model: {e}"); exit()

        constraints = {
            'age': (20, 85),              
            'height': (150, 190),        
            'weight': (45, 110),
            'waistline': (60, 120),
            'sight_left': (0, 1.7),
            'sight_right': (0, 1.7),
            'hear_left': ['1', '2'],
            'hear_right': ['1', '2'],
            'SBP': (80, 180),
            'DBP': (40, 110),
            'BLDS': (60, 150),
            'tot_chole': (90, 330),
            'HDL_chole': (20, 110),
            'LDL_chole': (20, 240),
            'triglyceride': (20, 430),
            'hemoglobin': (8, 18),
            'serum_creatinine': (0.4, 1.5),
            'SGOT_AST': (7, 60),
            'SGOT_ALT': (2, 85),
            'gamma_GTP': (5, 130),    
            'SMK_stat_type_cd': ['1', '2', '3'] # Allow non-smoker (3) too now
        }
        '''
        # --- Generate Constrained Synthetic Data (Example 1 - Broader) ---
        print("\nGenerating synthetic data with constraints (Broader Settings - Example 1)...")
        num_generate_constrained = 200 # More samples
        num_seeds = 5 # <<< USE MULTIPLE SEEDS
        all_constrained_dfs = []

        if len(scaled_data) >= generator.seq_length + num_seeds: # Ensure enough data for diverse seeds
            max_seed_start_index = len(scaled_data) - generator.seq_length
            seed_indices = np.random.choice(max_seed_start_index, num_seeds, replace=False)

            for i, start_idx in enumerate(seed_indices):
                print(f"  Generating batch {i+1}/{num_seeds} using seed from index {start_idx}...")
                seed = torch.FloatTensor(scaled_data[start_idx : start_idx + generator.seq_length])
                # Can override instance defaults per-call if needed:
                constrained_df_batch = generator.generate_synthetic_data(
                    seed_data=seed,
                    num_samples=num_generate_constrained // num_seeds, # Generate portion per seed
                    value_constraints=constraints,
                    # generation_noise_factor=0.08, # Example: Override noise for this call
                    # categorical_sampling_method='nearest', # Example: Override sampling for this call
                    analyze_ranges=False # Analyze combined data later
                )
                # Keep only generated part (optional, depends if you want seed overlap)
                all_constrained_dfs.append(constrained_df_batch.iloc[generator.seq_length:])

            if all_constrained_dfs:
                final_constrained_df = pd.concat(all_constrained_dfs).reset_index(drop=True)
                print("\n--- Analysis of COMBINED Constrained Data Ranges (Multiple Seeds) ---")
                final_ranges = generator.analyze_generated_data_ranges(final_constrained_df)
                generator.print_ranges(final_ranges)
                print("----------------------------------------------------------------------\n")

                final_constrained_df.to_csv(constrained_synthetic_data_broad, index=False, sep=';')
                print(f"Combined constrained synthetic data saved to {constrained_synthetic_data_broad}")
                # Verification checks...
            else:
                print("No constrained data generated.")

        else:
            print(f"Skipping constrained generation: Not enough data for {num_seeds} diverse seeds.")

        '''
        # --- Generate Balanced & Constrained Synthetic Data (Example 2 - Broader) ---
        print(f"\nGenerating balanced & constrained synthetic data for '{target_class_column}' (Broader Settings - Example 2)...")
        target_classes = generator.categorical_mappings.get(target_class_column, {}).values()
        if not target_classes: print(f"Error: Could not find class values for '{target_class_column}'.")
        else:
            samples_needed = {'0': 1281, '1': 1339}
            valid_samples_per_class = {k: v for k, v in samples_needed.items() if k in target_classes}
            if not valid_samples_per_class: print(f"Warning: Specified class values don't match loaded classes.")
            elif len(scaled_data) >= generator.seq_length: # Only need one seed here as balancing handles diversity across classes
                seed_idx = np.random.randint(0, len(scaled_data) - generator.seq_length)
                print(f"  Using seed from index {seed_idx} for balancing...")
                seed = torch.FloatTensor(scaled_data[seed_idx : seed_idx + generator.seq_length])
                balanced_constrained_data = generator.generate_synthetic_data_balancing_classes(
                    seed_data=seed, class_column=target_class_column,
                    samples_per_class=valid_samples_per_class,
                    value_constraints=constraints,
                    # Uses instance defaults for noise/sampling unless overridden:
                    # generation_noise_factor=0.08,
                    # categorical_sampling_method='probabilistic',
                    analyze_ranges_per_class=False
                )
                if not balanced_constrained_data.empty:
                    balanced_constrained_data.to_csv(balanced_constrained_synthetic_data_broad, index=False, sep=';')
                    print(f"Balanced & constrained synthetic data saved to {balanced_constrained_synthetic_data_broad}")
                    # Verification checks...
                else: print("No balanced constrained data was generated.")
            else: print(f"Skipping balanced constrained generation: Not enough data for seed.")

        print("\nScript finished.")