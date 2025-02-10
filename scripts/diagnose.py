import os
import onnx
import numpy as np
import onnxruntime as ort
import torch
from scsims import SIMS


def extract_intermediate_output(onnx_model, input_tensor, target_output_name):
    """
    Given an ONNX model, an input tensor, and the name of a node's output,
    modify the ONNX model to include that node output (if not already present),
    run inference using ONNX Runtime, and return the output tensor from that node.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        input_tensor (torch.Tensor): The input to the model.
        target_output_name (str): The name of the node's output to extract.

    Returns:
        numpy.ndarray: The extracted intermediate output.
    """
    # Check if target_output_name is already an output; if not, try to add it from value_info.
    output_names = [out.name for out in onnx_model.graph.output]
    if target_output_name not in output_names:
        # Use shape inference to get value_info for intermediate tensors.
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        candidate = None
        for value_info in inferred_model.graph.value_info:
            if value_info.name == target_output_name:
                candidate = value_info
                break
        if candidate is None:
            raise Exception(
                f"Output name '{target_output_name}' not found in inferred model value_info."
            )
        # Add the candidate as a new model output.
        onnx_model.graph.output.extend([candidate])

    # Save a temporary ONNX model file with the updated outputs.
    tmp_model_path = "data/scratch/temp.onnx"
    onnx.save(onnx_model, tmp_model_path)

    # Create an ONNX Runtime session with the temporary model file.
    sess = ort.InferenceSession(tmp_model_path)

    # Prepare input: assume the first input of the ONNX model.
    input_name = sess.get_inputs()[0].name
    onnx_inputs = {input_name: input_tensor.cpu().numpy()}

    # Run inference to obtain all outputs.
    outputs = sess.run(None, onnx_inputs)

    # Locate and return the output corresponding to target_output_name.
    extracted_output = None
    for idx, out in enumerate(sess.get_outputs()):
        if out.name == target_output_name:
            extracted_output = outputs[idx]
            break

    os.remove(tmp_model_path)

    if extracted_output is None:
        raise Exception(f"Output '{target_output_name}' not found after inference.")

    return extracted_output


def close(msg, a, b):
    if type(b) == torch.Tensor:
        b = b.detach().numpy()
    if np.allclose(
        a,
        b,
        rtol=1e-3,
        atol=1e-4,
    ):
        print(f"{msg} Match")
    else:
        print(f"\033[31m{msg} Mismatch\033[0m")


def extract_intermediate_output_pytorch(model, input_tensor, target_module_path):
    """
    Given a PyTorch model, an input tensor, and the "path" (i.e. the named module)
    within the model, run the model and return the intermediate output produced
    by that module.
    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): The input tensor for the model.
        target_module_path (str): The full name of the module (as in model.named_modules())
                                   whose output will be captured.
    Returns:
        torch.Tensor: The output from the target module.
    """
    hook_outputs = []

    def hook_fn(module, input, output):
        # Append the captured output; using a list ensures we catch even if the hook is
        # called multiple times (e.g. for repeated use in a forward pass).
        hook_outputs.append(output)

    # Search for the module matching the target path.
    target_module = None
    for name, module in model.named_modules():
        if name == target_module_path:
            # print(f"{name} hooked")
            target_module = module
            break
    if target_module is None:
        raise Exception(f"Module with path '{target_module_path}' not found in model.")
    # Register the forward hook on the target module.
    handle = target_module.register_forward_hook(hook_fn)
    # Run a forward pass.
    _ = model(input_tensor)
    # Remove the hook.
    handle.remove()
    if not hook_outputs:
        raise Exception(
            f"Hook failed to capture any output from module '{target_module_path}'."
        )

    return hook_outputs


if __name__ == "__main__":

    sims = SIMS(
        weights_path="checkpoints/default.ckpt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()
    sims_model = sims.model

    onnx_model = onnx.load("public/models/default.onnx")

    num_samples = 10

    _ = torch.manual_seed(42)
    input_tensor = torch.randn(num_samples, len(sims.model.genes))
    input_tensor_norm = torch.nn.functional.normalize(input_tensor, dim=1)

    for i in range(num_samples):
        print(f"Sample {i}")

        a = extract_intermediate_output(onnx_model, input_tensor[i : i + 1], "lpnorm")
        b = input_tensor_norm[i : i + 1]
        close("LpNorm", a, b)

        a = extract_intermediate_output(
            onnx_model,
            input_tensor[i : i + 1],
            "/network/tabnet/encoder/initial_bn/BatchNormalization_output_0",
        )
        b = extract_intermediate_output_pytorch(
            sims_model,
            input_tensor_norm[i : i + 1],
            "network.tabnet.encoder.initial_bn",
        )[0]
        close("Initial Batchnorm", a, b)

        # ReLu before final FC output layer
        b = extract_intermediate_output_pytorch(
            sims_model, input_tensor_norm[i : i + 1], "network.tabnet.encoder.relu"
        )

        a = extract_intermediate_output(
            onnx_model, input_tensor[i : i + 1], "/network/tabnet/encoder/Relu_output_0"
        )
        close("Relu 0", a, b[0])

        a = extract_intermediate_output(
            onnx_model,
            input_tensor[i : i + 1],
            "/network/tabnet/encoder/Relu_1_output_0",
        )
        close("Relu 1", a, b[1])

        a = extract_intermediate_output(
            onnx_model,
            input_tensor[i : i + 1],
            "/network/tabnet/encoder/Relu_2_output_0",
        )
        close("Relu 2", a, b[2])

        # Final Logits
        a = extract_intermediate_output(
            onnx_model,
            input_tensor[i : i + 1],
            "/network/tabnet/final_mapping/MatMul_output_0",
        )
        b = extract_intermediate_output_pytorch(
            sims_model, input_tensor_norm[i : i + 1], "network.tabnet.final_mapping"
        )[0][0]
        close("Final Mapping", a, b)
