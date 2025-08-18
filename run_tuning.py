from tuning import tune_hyperparameters

if __name__ == "__main__":
    # Define the hyperparameter grid
    param_grid = {
        'd_model': [32, 64, 128],
        'nhead': [4, 8, 16],
        'num_encoder_layers': [2, 3],
        'dim_feedforward': [64, 128, 256],
        'dropout': [0.1, 0.2],
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'max_seq_length': [50]
    }
    
    # Specify the CSV file and any categorical columns, e.g., ["CategoryColumnName"]
    csv_path = "balanced_Dataset_cleaned.csv"
    categorical_columns = ["sex","hear_left", "hear_right","SMK_stat_type_cd","DRK_YN"]
    
    best_params, tuning_history = tune_hyperparameters(csv_path, param_grid, seq_length=50, tuning_epochs=10, categorical_columns=categorical_columns)
