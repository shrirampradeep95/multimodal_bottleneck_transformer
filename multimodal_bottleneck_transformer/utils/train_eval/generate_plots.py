import matplotlib.pyplot as plt
import os
import csv
from utils.train_eval.attention_rollout import AttentionRolloutVisualizer


class TrainingPlotter:
    def __init__(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.train_losses = []
        self.eval_scores = []

    def log(self, train_loss, eval_score):
        self.train_losses.append(train_loss)
        self.eval_scores.append(eval_score)

    def plot(self, parameters, dataset="audioset", model=None, test_dataset=None, device="cuda"):
        # Save training history to CSV
        videos_to_use = parameters["videos_to_use"]
        modality = parameters["modality"]
        epochs = parameters["epochs"]

        csv_path = os.path.join(self.save_dir, f"training_log_{dataset}_{modality}_{epochs}_{videos_to_use}.csv")
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

        plt.figure(figsize=(10, 4))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.grid(True)
        plt.legend()

        # Plot eval score
        plt.subplot(1, 2, 2)
        if dataset == "vggsound":
            top1 = [s[0] for s in self.eval_scores]
            top5 = [s[1] for s in self.eval_scores]
            plt.plot(top1, label="Top-1 Accuracy", marker="o")
            plt.plot(top5, label="Top-5 Accuracy", marker="x")
            plt.ylabel("Accuracy")
        else:
            plt.plot(self.eval_scores, label="mAP", marker="x")
            plt.ylabel("mAP")

        plt.xlabel("Epoch")
        plt.title("Evaluation Score over Epochs")
        plt.grid(True)
        plt.legend()

        # Save plot
        save_path = os.path.join(self.save_dir, f"training_plot_{dataset}_{modality}_{epochs}_{videos_to_use}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved training plot to {save_path}")

        # Optional: visualize attention maps for AudioSet
        # if dataset == "audioset" and model is not None and test_dataset is not None:
        #     visualizer = AttentionRolloutVisualizer(model, test_dataset, device=device)
        #     visualizer.visualize(num_samples=3)
