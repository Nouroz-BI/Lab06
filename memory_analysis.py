
import json
import csv
from pathlib import Path

import torch
import lab06_starter as S


# --------------------------------------------------
# Part 4: Memory Analysis
# --------------------------------------------------

# Use the same answer-only loss from Part 1.
def answer_only_labels(input_ids, attention_mask, prompt_lens):
    labels = torch.full_like(input_ids, -100)

    rows = torch.arange(input_ids.size(0))
    answer_positions = prompt_lens.long()

    labels[rows, answer_positions] = input_ids[rows, answer_positions]

    return labels


def bytes_to_mib(value):
    return value / (1024 ** 2)


def calculate_memory(model):
    """
    Calculate theoretical memory for:
    weights, gradients and Adam optimizer states.

    Model uses float32, so each parameter uses 4 bytes.
    Adam uses two states of 4 bytes each.
    """

    trainable, total = S.count_params(model)

    bytes_per_parameter = 4

    predicted_weights = total * bytes_per_parameter
    predicted_gradients = trainable * bytes_per_parameter
    predicted_optimizer = trainable * 2 * bytes_per_parameter

    predicted_total = (
        predicted_weights
        + predicted_gradients
        + predicted_optimizer
    )

    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "predicted_weights_bytes": predicted_weights,
        "predicted_gradients_bytes": predicted_gradients,
        "predicted_optimizer_bytes": predicted_optimizer,
        "predicted_training_state_bytes": predicted_total
    }


def run_memory_experiment(mode, seed, train_examples, tokenizer):
    """
    Run a short training experiment to measure memory.

    Five steps are enough to initialize Adam's states
    and collect the training-state measurements.
    """

    torch.manual_seed(seed)

    model, tok = S.load_model()

    if mode == "lora":
        model = S.apply_lora(model, r=16)

    theoretical = calculate_memory(model)

    print("\n" + "=" * 55)
    print("Experiment:", mode.upper())
    print("=" * 55)

    # Train for five steps to initialize gradients and Adam states.
    actual = S.train(
        model,
        tok,
        train_examples,
        steps=5,
        lr=1e-4 if mode == "full" else 0.001,
        batch_size=8,
        seed=seed,
        labels_fn=answer_only_labels
    )

    measured_weights = actual["weight_bytes"]
    measured_gradients = actual["grad_bytes"]
    measured_optimizer = actual["optimizer_bytes"]

    measured_total = (
        measured_weights
        + measured_gradients
        + measured_optimizer
    )

    result = {
        "method": mode,
        **theoretical,
        "measured_weights_bytes": measured_weights,
        "measured_gradients_bytes": measured_gradients,
        "measured_optimizer_bytes": measured_optimizer,
        "measured_training_state_bytes": measured_total,
        "training_seconds": actual["seconds"],
        "peak_cuda_bytes": actual["peak_cuda_bytes"]
    }

    print("Trainable parameters:", f'{theoretical["trainable_parameters"]:,}')
    print("Total parameters:", f'{theoretical["total_parameters"]:,}')

    print("\nPredicted memory (MiB)")
    print("Weights:", round(bytes_to_mib(
        theoretical["predicted_weights_bytes"]), 2))
    print("Gradients:", round(bytes_to_mib(
        theoretical["predicted_gradients_bytes"]), 2))
    print("Adam states:", round(bytes_to_mib(
        theoretical["predicted_optimizer_bytes"]), 2))

    print("\nMeasured memory (MiB)")
    print("Weights:", round(bytes_to_mib(measured_weights), 2))
    print("Gradients:", round(bytes_to_mib(measured_gradients), 2))
    print("Adam states:", round(bytes_to_mib(measured_optimizer), 2))
    print("Total:", round(bytes_to_mib(measured_total), 2))

    print("\nTraining time:", round(actual["seconds"], 2), "seconds")

    if actual["peak_cuda_bytes"] is None:
        print("GPU memory: Not available (CPU-only environment)")
    else:
        print(
            "Peak GPU memory:",
            round(bytes_to_mib(actual["peak_cuda_bytes"]), 2),
            "MiB"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():

    print("PART 4: MEMORY ANALYSIS")
    print("Device:", S.DEVICE)

    # Load the same dataset and sample size used in the lab.
    model, tokenizer = S.load_model()
    train, _, _ = S.load_task(
        "emotion",
        n_train=600,
        n_test=500,
        data_seed=0,
        tokenizer=tokenizer
    )
    del model

    results = []

    # Full fine-tuning memory measurement.
    results.append(
        run_memory_experiment("full", 0, train, tokenizer)
    )

    # LoRA rank 16 memory measurement.
    results.append(
        run_memory_experiment("lora", 0, train, tokenizer)
    )

    # Save results.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "memory_analysis.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(output_dir / "memory_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("PART 4 COMPLETED")
    print("=" * 55)
    print("Saved: results/memory_analysis.json")
    print("Saved: results/memory_analysis.csv")


if __name__ == "__main__":
    main()
