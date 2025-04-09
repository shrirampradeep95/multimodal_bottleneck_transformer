import matplotlib.pyplot as plt
import os
import csv


class TrainingPlotter:
    """
    Class to log and visualize training loss and evaluation metrics over epochs.
    Also saves training logs to CSV and plots to PNG.
    """
    def __init__(self, save_dir="plots"):
        """
        Initializes the plotter.
        Args:
            save_dir: Directory where plots and logs will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.train_losses = []
        self.eval_scores = []

    def log(self, train_loss, eval_score):
        """
        Logs the training loss and evaluation score after each epoch.
        Args:
            train_loss: Training loss for the epoch.
            eval_score: Evaluation score
        """
        self.train_losses.append(train_loss)
        self.eval_scores.append(eval_score)

    def plot(self, parameters, dataset):
        """
        Plots training loss and evaluation metrics and saves them to files.
        Args:
            parameters: Configuration dictionary used during training.
        """
        # Extract details from config for naming
        videos_to_use = parameters["videos_to_use"]
        modality = parameters["modality"]
        epochs = parameters["epochs"]

        # Create path for the CSV log
        csv_path = os.path.join(
            self.save_dir, f"training_log_{dataset}_{modality}_{epochs}_{videos_to_use}.csv"
        )

        # Save training logs to CSV file
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            if dataset == "vggsound":
                writer.writerow(["Epoch", "Train Loss", "Top-1 Acc", "Top-5 Acc"])
                for i, (loss, score) in enumerate(zip(self.train_losses, self.eval_scores), 1):
                    writer.writerow([i, loss, score[0], score[1]])
            else:
                writer.writerow(["Epoch", "Train Loss", "mAP"])
                for i, (loss, score) in enumerate(zip(self.train_losses, self.eval_scores), 1):
                    writer.writerow([i, loss, score])
        print(f"Saved training log to {csv_path}")

        # Start plotting
        plt.figure(figsize=(10, 4))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.grid(True)
        plt.legend()

        # Plot evaluation metric
        plt.subplot(1, 2, 2)
        if dataset == "vggsound":
            # For single-label classification, plot Top-1 and Top-5 accuracy
            top1 = [s[0] for s in self.eval_scores]
            top5 = [s[1] for s in self.eval_scores]
            plt.plot(top1, label="Top-1 Accuracy", marker="o")
            plt.plot(top5, label="Top-5 Accuracy", marker="x")
            plt.ylabel("Accuracy")
        else:
            # For multi-label classification, plot mAP
            plt.plot(self.eval_scores, label="mAP", marker="x")
            plt.ylabel("mAP")

        plt.xlabel("Epoch")
        plt.title("Evaluation Score over Epochs")
        plt.grid(True)
        plt.legend()

        # Save the plot image
        save_path = os.path.join(
            self.save_dir, f"training_plot_{dataset}_{modality}_{epochs}_{videos_to_use}.png"
        )
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved training plot to {save_path}")
