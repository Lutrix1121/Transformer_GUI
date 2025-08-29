import os
import matplotlib.pyplot as plt

def save_training_plots(history: dict, output_dir: str, combo_idx: int):
        """
        Save training/validation loss and R² plots for a given training history.

        Args:
            history: Dictionary containing 'train_loss', 'val_loss', 'val_r2'
            params: Hyperparameters used for this run
            output_dir: Directory where plots should be saved
            combo_idx: Index of the parameter combination
        """
        if not history:
            return

        os.makedirs(output_dir, exist_ok=True)
        epochs = range(1, len(history.get('train_loss', [])) + 1)

        # Plot Loss
        plt.figure()
        if 'train_loss' in history:
            plt.plot(epochs, history['train_loss'], label="Train Loss")
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve (Combination {combo_idx})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"loss_curve_{combo_idx}.png"))
        plt.close()

        if 'val_r2' in history and history['val_r2']:
            plt.figure()
            plt.plot(epochs, history['val_r2'], label="Validation R²")
            plt.xlabel("Epoch")
            plt.ylabel("R²")
            plt.title(f"Validation R² (Combination {combo_idx})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"val_r2_{combo_idx}.png"))
            plt.close()