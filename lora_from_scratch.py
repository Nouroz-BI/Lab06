
import copy
import torch
import torch.nn as nn
import lab06_starter as S

from peft import LoraConfig, get_peft_model


# --------------------------------------------------
# 1. Create LoRA from scratch
# --------------------------------------------------

class LoRALinear(nn.Module):

    def __init__(self, base, r, alpha):

        super().__init__()

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the original linear layer
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # Create the two LoRA matrices
        self.A = nn.Parameter(
            torch.empty(
                r,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                base.out_features,
                r,
                device=base.weight.device,
                dtype=base.weight.dtype
            )
        )

        # Initialize A with small random values
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    def forward(self, x):

        # Original linear layer output
        original_output = self.base(x)

        # LoRA update
        lora_output = x @ self.A.T @ self.B.T

        return original_output + self.scaling * lora_output


# --------------------------------------------------
# 2. Test the initial output and parameter count
# --------------------------------------------------

def test_initial_output_and_parameters():

    print("\nTest 1: Initial Output and Parameter Count")
    print("------------------------------------------")

    torch.manual_seed(42)

    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)

    rank = 2
    alpha = 4

    lora = LoRALinear(base, rank, alpha)

    # At initialization, B is zero.
    # Therefore, LoRA output should equal the original output.
    original_output = base(x)
    lora_output = lora(x)

    torch.testing.assert_close(
        lora_output,
        original_output
    )

    print("Initial output check passed.")

    # Check the number of trainable parameters
    trainable_parameters = sum(
        p.numel()
        for p in lora.parameters()
        if p.requires_grad
    )

    expected_parameters = rank * (
        base.in_features + base.out_features
    )

    assert trainable_parameters == expected_parameters

    print("Trainable parameters:", trainable_parameters)
    print("Expected parameters:", expected_parameters)
    print("Parameter count check passed.")


# --------------------------------------------------
# 3. Test merging LoRA weights into the base layer
# --------------------------------------------------

def test_merged_weights():

    print("\nTest 2: Merged Weights")
    print("----------------------")

    torch.manual_seed(42)

    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)

    rank = 2
    alpha = 4

    lora = LoRALinear(base, rank, alpha)

    # Give B non-zero values so the LoRA update is active
    nn.init.normal_(lora.B, mean=0.0, std=0.02)

    # Output using the separate LoRA matrices
    lora_output = lora(x)

    # Create a new linear layer for merged weights
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None
    )

    # Merge the LoRA update into the original weight
    with torch.no_grad():

        merged.weight.copy_(
            base.weight + lora.scaling * (lora.B @ lora.A)
        )

        if base.bias is not None:
            merged.bias.copy_(base.bias)

    merged_output = merged(x)

    # Both methods should produce the same output
    torch.testing.assert_close(
        lora_output,
        merged_output,
        rtol=1e-5,
        atol=1e-6
    )

    print("Merged weight check passed.")


# --------------------------------------------------
# 4. Compare our implementation with PEFT
# --------------------------------------------------

class TinyModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.q_proj = nn.Linear(8, 6)
        self.v_proj = nn.Linear(8, 6)

    def forward(self, x):

        return self.q_proj(x), self.v_proj(x)


def test_against_peft():

    print("\nTest 3: Compare with PEFT")
    print("-------------------------")

    torch.manual_seed(42)

    rank = 2
    alpha = 4

    # Create a small model for testing
    original_model = TinyModel()

    # Create our own LoRA version
    custom_model = copy.deepcopy(original_model)

    custom_model.q_proj = LoRALinear(
        custom_model.q_proj,
        rank,
        alpha
    )

    custom_model.v_proj = LoRALinear(
        custom_model.v_proj,
        rank,
        alpha
    )

    # Set B to non-zero values
    nn.init.normal_(
        custom_model.q_proj.B,
        mean=0.0,
        std=0.02
    )

    nn.init.normal_(
        custom_model.v_proj.B,
        mean=0.0,
        std=0.02
    )

    # Create a separate model using PEFT
    peft_base = copy.deepcopy(original_model)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none"
    )

    peft_model = get_peft_model(peft_base, config)

    # Copy our LoRA matrices into the PEFT layers
    peft_layers = peft_model.base_model.model

    with torch.no_grad():

        peft_layers.q_proj.lora_A["default"].weight.copy_(
            custom_model.q_proj.A
        )

        peft_layers.q_proj.lora_B["default"].weight.copy_(
            custom_model.q_proj.B
        )

        peft_layers.v_proj.lora_A["default"].weight.copy_(
            custom_model.v_proj.A
        )

        peft_layers.v_proj.lora_B["default"].weight.copy_(
            custom_model.v_proj.B
        )

    # Compare both models using the same input
    x = torch.randn(4, 8)

    custom_q, custom_v = custom_model(x)
    peft_q, peft_v = peft_model(x)

    torch.testing.assert_close(
        custom_q,
        peft_q,
        rtol=1e-5,
        atol=1e-6
    )

    torch.testing.assert_close(
        custom_v,
        peft_v,
        rtol=1e-5,
        atol=1e-6
    )

    print("q_proj comparison passed.")
    print("v_proj comparison passed.")
    print("Our implementation agrees with PEFT.")


# --------------------------------------------------
# 5. Count LoRA parameters in the actual model
# --------------------------------------------------

def show_model_parameter_counts():

    print("\nTest 4: LoRA Parameter Counts")
    print("------------------------------")

    # Load the model from the starter file
    model, tokenizer = S.load_model()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total model parameters:", total_parameters)

    # Find all q_proj and v_proj layers
    projection_layers = {
        "q_proj": [],
        "v_proj": []
    }

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            if name.endswith("q_proj"):
                projection_layers["q_proj"].append(module)

            elif name.endswith("v_proj"):
                projection_layers["v_proj"].append(module)

    # Calculate the LoRA parameters for each rank
    for projection_name, layers in projection_layers.items():

        print(f"\n{projection_name}")
        print("Number of layers:", len(layers))

        for rank in [1, 4, 16]:

            trainable_parameters = 0

            for layer in layers:

                trainable_parameters += rank * (
                    layer.in_features + layer.out_features
                )

            percentage = (
                trainable_parameters / total_parameters
            ) * 100

            print(
                f"Rank {rank}: "
                f"{trainable_parameters:,} trainable parameters "
                f"({percentage:.4f}% of the whole model)"
            )

    # The model is no longer needed
    del model


# --------------------------------------------------
# 6. Run all tests
# --------------------------------------------------

if __name__ == "__main__":

    test_initial_output_and_parameters()

    test_merged_weights()

    test_against_peft()

    show_model_parameter_counts()

    print("\nAll Part 2 checks completed.")
