
import torch
import lab06_starter as S


# Only calculate loss for the answer token
def answer_only_labels(input_ids, attention_mask, prompt_lens):

    # Set all labels to -100 so they are ignored
    labels = torch.full_like(input_ids, -100)

    batch_size = input_ids.size(0)
    rows = torch.arange(batch_size, device=input_ids.device)


    # Find the position of the answer token
    answer_positions = prompt_lens.to(input_ids.device).long()

    # Check that the answer position is inside the sequence
    assert torch.all(answer_positions < input_ids.size(1))

    # Check that the answer token is not padding
    assert torch.all(attention_mask[rows, answer_positions] == 1)

    # Include only the answer token in the loss
    labels[rows, answer_positions] = input_ids[rows, answer_positions]

    return labels


# Check that answer-only labels work correctly
def verify_answer_only_loss(train, tokenizer):

    # Select four examples for testing
    batch = train[:4]

    input_ids, attention_mask, prompt_lens = S.pad_batch(
        batch,
        tokenizer.pad_token_id
    )

    labels = answer_only_labels(
        input_ids,
        attention_mask,
        prompt_lens
    )

    # Each example should have exactly one scored token
    scored_counts = (labels != -100).sum(dim=1)

    assert torch.all(scored_counts == 1)

    # Check that prompt tokens are ignored
    for i in range(input_ids.size(0)):

        prompt_length = prompt_lens[i].item()

        assert torch.all(labels[i, :prompt_length] == -100)

        # Check that the answer token is included
        assert labels[i, prompt_length] == input_ids[i, prompt_length]

    print("Answer-only loss checks passed.")
    print("Number of examples checked:", input_ids.size(0))
    print("Scored tokens per example:", scored_counts.tolist())


# Run the zero-shot baseline
def run_zero_shot_baseline(model, tokenizer, test, label_ids):

    # Evaluate the model without training
    accuracy = S.evaluate(model, tokenizer, test, label_ids)

    # There are six emotion classes
    chance_accuracy = 1 / 6

    print("\nZero-Shot Baseline Results")
    print("--------------------------")
    print(f"Zero-shot accuracy: {accuracy:.4f}")
    print(f"Random guessing accuracy: {chance_accuracy:.4f}")
    print(f"Difference: {accuracy - chance_accuracy:.4f}")

    return accuracy


if __name__ == "__main__":

    # Load the model and tokenizer
    model, tokenizer = S.load_model()

    # Load the emotion dataset
    train, test, label_ids = S.load_task(
        "emotion",
        n_train=600,
        n_test=500,
        tokenizer=tokenizer
    )

    # Check the answer-only loss
    verify_answer_only_loss(train, tokenizer)

    # Measure accuracy before training
    baseline_accuracy = run_zero_shot_baseline(
        model,
        tokenizer,
        test,
        label_ids
    )
