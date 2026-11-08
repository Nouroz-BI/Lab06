
import csv
import json
import os
import statistics
import torch
import matplotlib.pyplot as plt

import lab06_starter as S


# --------------------------------------------------
# Lab 06 - Part 3: Training Experiments
# --------------------------------------------------

# Experiment settings
N_TRAIN = 600
N_TEST = 500
DATA_SEED = 0

SEEDS = [0, 1, 2]
RANKS = [1, 4, 16]

STEPS = 120
BATCH_SIZE = 8

# Learning rates to compare using validation data
FULL_LR_CANDIDATES = [1e-5, 5e-5, 1e-4]
LORA_LR_CANDIDATES = [3e-4, 1e-3, 3e-3]

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Answer-only loss
# --------------------------------------------------

def answer_only_labels(input_ids, attention_mask, prompt_lens):

    # Ignore every token initially
    labels = torch.full_like(input_ids, -100)

    batch_size = input_ids.size(0)
    rows = torch.arange(batch_size, device=input_ids.device)

    # Find the answer token positions
    answer_positions = prompt_lens.to(input_ids.device).long()

    # Make sure answer positions are valid
    assert torch.all(answer_positions < input_ids.size(1))

    # Make sure answer tokens are not padding
    assert torch.all(
        attention_mask[rows, answer_positions] == 1
    )

    # Calculate loss only for the answer token
    labels[rows, answer_positions] = (
        input_ids[rows, answer_positions]
    )

    return labels


# --------------------------------------------------
# 2. Load training, validation and test data
# --------------------------------------------------


# Load the tokenizer once
print("\nLoading tokenizer and dataset...")

_, tokenizer = S.load_model()

# Load training and validation data
train, validation, label_ids = S.load_task(
    "emotion",
    n_train=N_TRAIN,
    n_test=N_TEST,
    data_seed=DATA_SEED,
    tokenizer=tokenizer,
    eval_split="validation"
)

# Load test data using the same tokenizer
_, test, test_label_ids = S.load_task(
    "emotion",
    n_train=N_TRAIN,
    n_test=N_TEST,
    data_seed=DATA_SEED,
    tokenizer=tokenizer,
    eval_split="test"
)

# Check that both dataset splits use the same labels
assert label_ids == test_label_ids

print("Training examples:", len(train))
print("Validation examples:", len(validation))
print("Test examples:", len(test))


train, validation, label_ids = S.load_task(
    "emotion",
    n_train=N_TRAIN,
    n_test=N_TEST,
    data_seed=DATA_SEED,
    tokenizer=S.load_model()[1],
    eval_split="validation"
)

# Load the test split separately
_, test, test_label_ids = S.load_task(
    "emotion",
    n_train=N_TRAIN,
    n_test=N_TEST,
    data_seed=DATA_SEED,
    tokenizer=S.load_model()[1],
    eval_split="test"
)

assert label_ids == test_label_ids

print("Training examples:", len(train))
print("Validation examples:", len(validation))
print("Test examples:", len(test))


# --------------------------------------------------
# 3. Write the prediction before training
# --------------------------------------------------

print("\nExperiment Prediction")
print("---------------------")

prediction = (
    "I expect rank 16 to perform better than rank 1 because "
    "it has more trainable parameters. I expect rank 16 to "
    "approach full fine-tuning, but it may not match it. "
    "Full fine-tuning can update all model parameters, while "
    "LoRA updates only the selected adapter parameters."
)

print(prediction)


# --------------------------------------------------
# 4. Helper function to run one experiment
# --------------------------------------------------

def run_experiment(
    mode,
    seed,
    evaluation_data,
    rank=8,
    learning_rate=None,
    labels_fn=answer_only_labels
):

    result = S.run(
        mode=mode,
        seed=seed,
        train_examples=train,
        test=evaluation_data,
        label_ids=label_ids,
        r=rank,
        steps=STEPS,
        lr=learning_rate,
        labels_fn=labels_fn,
        batch_size=BATCH_SIZE
    )

    return result


# --------------------------------------------------
# 5. Select learning rates using validation data
# --------------------------------------------------

def select_learning_rate(mode, candidates, rank=16):

    print("\nLearning Rate Selection")
    print("-----------------------")
    print("Method:", mode)

    validation_results = []

    for learning_rate in candidates:

        print(
            f"\nTesting learning rate: {learning_rate}"
        )

        result = run_experiment(
            mode=mode,
            seed=0,
            evaluation_data=validation,
            rank=rank,
            learning_rate=learning_rate,
            labels_fn=answer_only_labels
        )

        accuracy = float(result["accuracy"])

        print("Validation accuracy:", accuracy)

        validation_results.append({
            "mode": mode,
            "rank": rank,
            "learning_rate": learning_rate,
            "validation_accuracy": accuracy
        })

    # Select the learning rate with the highest validation accuracy.
    # If tied, choose the smaller learning rate.
    best_result = max(
        validation_results,
        key=lambda x: (
            x["validation_accuracy"],
            -x["learning_rate"]
        )
    )

    print("\nSelected learning rate:", best_result["learning_rate"])
    print(
        "Best validation accuracy:",
        best_result["validation_accuracy"]
    )

    return best_result["learning_rate"], validation_results


# --------------------------------------------------
# 6. Save experiment results
# --------------------------------------------------

def save_results(results, filename):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8") as file:

        if not results:
            return

        # Collect all columns from the result dictionaries
        columns = []

        for row in results:
            for key in row:
                if key not in columns:
                    columns.append(key)

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", path)


# --------------------------------------------------
# 7. Run the required training experiments
# --------------------------------------------------

def main():

    all_results = []
    validation_results = []

    # Select learning rates without using the test set
    full_lr, full_tuning = select_learning_rate(
        mode="full",
        candidates=FULL_LR_CANDIDATES
    )

    lora_lr, lora_tuning = select_learning_rate(
        mode="lora",
        candidates=LORA_LR_CANDIDATES,
        rank=16
    )

    validation_results.extend(full_tuning)
    validation_results.extend(lora_tuning)

    save_results(
        validation_results,
        "learning_rate_validation.csv"
    )

    print("\nSelected Learning Rates")
    print("-----------------------")
    print("Full fine-tuning:", full_lr)
    print("LoRA:", lora_lr)

    # ----------------------------------------------
    # Experiment A: Full fine-tuning
    # ----------------------------------------------

    print("\nExperiment A: Full Fine-Tuning")
    print("================================")

    for seed in SEEDS:

        print(f"\nFull fine-tuning - Seed {seed}")

        result = run_experiment(
            mode="full",
            seed=seed,
            evaluation_data=test,
            learning_rate=full_lr,
            labels_fn=answer_only_labels
        )

        row = {
            "method": "Full fine-tuning",
            "loss_type": "answer_only",
            "rank": 0,
            "seed": seed,
            "learning_rate": full_lr,
            **result
        }

        all_results.append(row)

        print("Test accuracy:", result["accuracy"])
        print("First loss:", result["first_loss"])
        print("Last loss:", result["last_loss"])
        print("Training time:", result["seconds"])

    # ----------------------------------------------
    # Experiment B: LoRA ranks 1, 4 and 16
    # Answer-only loss
    # ----------------------------------------------

    print("\nExperiment B: LoRA Rank Comparison")
    print("===================================")

    for rank in RANKS:

        for seed in SEEDS:

            print(
                f"\nLoRA Rank {rank} - Seed {seed}"
            )

            result = run_experiment(
                mode="lora",
                seed=seed,
                evaluation_data=test,
                rank=rank,
                learning_rate=lora_lr,
                labels_fn=answer_only_labels
            )

            row = {
                "method": "LoRA",
                "loss_type": "answer_only",
                "rank": rank,
                "seed": seed,
                "learning_rate": lora_lr,
                **result
            }

            all_results.append(row)

            print("Test accuracy:", result["accuracy"])
            print("First loss:", result["first_loss"])
            print("Last loss:", result["last_loss"])
            print("Training time:", result["seconds"])

    # ----------------------------------------------
    # Experiment C: Rank 16 with all-token loss
    # ----------------------------------------------

    print("\nExperiment C: Rank 16 with All-Token Loss")
    print("==========================================")

    for seed in SEEDS:

        print(f"\nLoRA Rank 16 All-Token - Seed {seed}")

        result = run_experiment(
            mode="lora",
            seed=seed,
            evaluation_data=test,
            rank=16,
            learning_rate=lora_lr,
            labels_fn=S.all_tokens_labels
        )

        row = {
            "method": "LoRA",
            "loss_type": "all_tokens",
            "rank": 16,
            "seed": seed,
            "learning_rate": lora_lr,
            **result
        }

        all_results.append(row)

        print("Test accuracy:", result["accuracy"])
        print("First loss:", result["first_loss"])
        print("Last loss:", result["last_loss"])
        print("Training time:", result["seconds"])

    # ----------------------------------------------
    # Save all experiment results
    # ----------------------------------------------

    save_results(
        all_results,
        "training_results.csv"
    )

    # Save complete results in JSON format
    json_path = os.path.join(
        OUTPUT_DIR,
        "training_results.json"
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            default=str
        )

    print("Saved:", json_path)

    # ----------------------------------------------
    # Summary by experiment
    # ----------------------------------------------

    print("\nExperiment Summary")
    print("==================")

    groups = {}

    for row in all_results:

        key = (
            row["method"],
            row["loss_type"],
            row["rank"]
        )

        groups.setdefault(key, []).append(
            float(row["accuracy"])
        )

    summary_rows = []

    for key, accuracies in groups.items():

        mean_accuracy = statistics.mean(accuracies)

        std_accuracy = (
            statistics.stdev(accuracies)
            if len(accuracies) > 1
            else 0.0
        )

        summary_row = {
            "method": key[0],
            "loss_type": key[1],
            "rank": key[2],
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "number_of_seeds": len(accuracies)
        }

        summary_rows.append(summary_row)

        print(
            f"{key}: "
            f"Mean accuracy = {mean_accuracy:.4f}, "
            f"Standard deviation = {std_accuracy:.4f}"
        )

    save_results(
        summary_rows,
        "experiment_summary.csv"
    )

    # ----------------------------------------------
    # Create accuracy comparison figure
    # ----------------------------------------------

    plot_labels = []
    plot_means = []
    plot_stds = []

    for row in summary_rows:

        label = (
            f"{row['method']}\n"
            f"{row['loss_type']}\n"
            f"Rank {row['rank']}"
        )

        plot_labels.append(label)
        plot_means.append(row["mean_accuracy"])
        plot_stds.append(row["std_accuracy"])

    plt.figure(figsize=(12, 6))

    plt.bar(
        plot_labels,
        plot_means,
        yerr=plot_stds,
        capsize=5
    )

    plt.axhline(
        y=1 / 6,
        linestyle="--",
        label="Random guessing (16.67%)"
    )

    plt.axhline(
        y=0.326,
        linestyle=":",
        label="Zero-shot baseline (32.60%)"
    )

    plt.ylabel("Test Accuracy")
    plt.xlabel("Training Method")
    plt.title("Lab 06 - Full Fine-Tuning and LoRA Comparison")

    plt.ylim(0, 1)
    plt.xticks(rotation=0, fontsize=8)
    plt.legend()
    plt.tight_layout()

    figure_path = os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("\nSaved figure:", figure_path)

    print("\nAll Part 3 experiments completed.")


# --------------------------------------------------
# Start the experiments
# --------------------------------------------------

if __name__ == "__main__":

    main()
