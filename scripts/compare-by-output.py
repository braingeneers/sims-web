"""
Compare pytorch vs. onnx intermediate outputs by looking for intermediate
outputs with matching shapes and values for a sample that has concordant
final output and then looking at those pairs with a sample that does not have
concordant outputs.
"""

import os
import copy
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchviz import make_dot
from scsims import SIMS


# ------------------------------
# Helper: Capture PyTorch intermediate outputs via hooks
# ------------------------------
def capture_pytorch_intermediates(model, input_tensor):
    """
    Registers forward hooks on all submodules in model and
    returns a dict mapping module full names -> output (first call).
    """
    intermediates = {}
    handles = []

    def get_hook(name):
        def hook(module, input, output):
            # If the output is a single Tensor, process it.
            if torch.is_tensor(output):
                intermediates[name] = output.detach().cpu().numpy()
            # If the output is a tuple or list, try converting each Tensor element.
            elif isinstance(output, (tuple, list)):
                converted = []
                for elem in output:
                    if torch.is_tensor(elem):
                        converted.append(elem.detach().cpu().numpy())
                    else:
                        converted.append(elem)
                intermediates[name] = converted
            else:
                # For any other type, print a warning.
                print(f"Skipping {name} output of type {type(output)}")

        return hook

    # Register hook on every module that is not a container
    for name, module in model.named_modules():
        # Skip the top-level module (empty name) if desired or modules without parameters
        if name != "":
            handles.append(module.register_forward_hook(get_hook(name)))
    # Run forward pass in eval mode
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    # Remove all hooks
    for h in handles:
        h.remove()

    return intermediates


# ------------------------------
# Helper: Capture ONNX intermediate outputs
# ------------------------------
def capture_onnx_intermediates(onnx_model, input_tensor):
    """
    Given an ONNX model, add all intermediate value_info outputs and run inference
    to capture intermediate outputs. Returns a dict mapping output names -> output np.array.
    """
    # Create a copy to avoid modifying the original model
    # Segfaults...
    # model_cp = copy.deepcopy(onnx_model)
    model_cp = onnx_model
    # Infer shapes so that value_info is populated
    inferred_model = onnx.shape_inference.infer_shapes(model_cp)

    # Add every intermediate tensor available in value_info as an output.
    existing_outputs = {out.name for out in inferred_model.graph.output}
    for val in inferred_model.graph.value_info:
        if val.name not in existing_outputs:
            inferred_model.graph.output.extend([val])
            existing_outputs.add(val.name)

    # Save temporary ONNX model with extra outputs.
    tmp_model_path = "data/validation/temp_intermediates.onnx"
    onnx.save(inferred_model, tmp_model_path)

    # Create an ONNX Runtime session
    ort_session = ort.InferenceSession(tmp_model_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor.cpu().numpy()}
    outputs = ort_session.run(None, ort_inputs)
    # Map each output name to its result.
    onnx_outputs = {}
    for idx, out_meta in enumerate(ort_session.get_outputs()):
        onnx_outputs[out_meta.name] = outputs[idx]
    os.remove(tmp_model_path)
    return onnx_outputs


# ------------------------------
# Helper: Compare outputs with tolerance.
# ------------------------------
def compare_intermediates(pt_intermediates, onnx_intermediates, rtol=1e-2, atol=1e-3):
    """
    Compare each PyTorch intermediate output with each ONNX output using np.allclose.
    Prints which pairs match.
    """
    matched = []
    for pt_name, pt_value in pt_intermediates.items():
        for onnx_name, onnx_value in onnx_intermediates.items():
            # print(f"Comparing: PyTorch [{pt_name}]  <-> ONNX [{onnx_name}]")
            # print(f"Types: {type(pt_value)}  <-> {type(onnx_value)}")
            # Skip if shapes differ
            if type(pt_value) != type(onnx_value):
                # print(f"Type mismatch: PyTorch [{pt_name}]  <-> ONNX [{onnx_name}]")
                continue
            if pt_value.shape != onnx_value.shape:
                continue
            if np.allclose(pt_value, onnx_value, rtol=rtol, atol=atol):
                # print(f"Match: PyTorch [{pt_name}]  <-> ONNX [{onnx_name}]")
                matched.append((pt_name, onnx_name))
    return matched


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Load the pytorch model from a checkpoint
    sims = SIMS(
        weights_path="checkpoints/default.ckpt",
        map_location=torch.device("cpu"),
    )
    model = sims.model.network
    _ = model.eval()

    # Create representative input tensors. Onnx has lpnorm built in so
    # create normalized and non-normalized versions
    num_samples = 10
    _ = torch.manual_seed(42)
    input_tensor = torch.randn(num_samples, len(sims.model.genes))
    input_tensor_norm = torch.nn.functional.normalize(input_tensor, dim=1)

    onnx_path = "public/models/default.onnx"
    onnx_model = onnx.load(onnx_path)

    # Get pairs from sample 0 which is known to generate concordant output
    # for the default model
    i = 0

    # Capture ONNX intermediate outputs.
    onnx_intermediates = capture_onnx_intermediates(onnx_model, input_tensor[i : i + 1])
    print(f"Captured {len(onnx_intermediates)} ONNX intermediate outputs.")

    # Capture PyTorch intermediate outputs.
    pt_intermediates = capture_pytorch_intermediates(
        model, input_tensor_norm[i : i + 1]
    )
    print(f"Captured {len(pt_intermediates)} PyTorch intermediate outputs.")

    # Compare them using np.allclose.
    matched = compare_intermediates(pt_intermediates, onnx_intermediates)
    print(f"Found {len(matched)} matching intermediate outputs.")

    for m in matched:
        if "final" in m[1]:
            print("Logits match")

    # Get pairs from a sample that differs the most from the compare-by-path
    i = 7

    # Capture ONNX intermediate outputs.
    onnx_intermediates = capture_onnx_intermediates(onnx_model, input_tensor[i : i + 1])
    print(f"Captured {len(onnx_intermediates)} ONNX intermediate outputs.")

    # Capture PyTorch intermediate outputs.
    pt_intermediates = capture_pytorch_intermediates(
        model, input_tensor_norm[i : i + 1]
    )
    print(f"Captured {len(pt_intermediates)} PyTorch intermediate outputs.")

    # Compare them using np.allclose.
    differ = compare_intermediates(pt_intermediates, onnx_intermediates)
    print(f"Found {len(matched)} matching intermediate outputs.")

    for m in differ:
        if "final" in m[1]:
            print("Logits match")

    diverge = [x for x in matched if x not in differ]

    # mismatched = set(matched) - set(differ)
    # print(f"Found {len(mismatched)} mismatched intermediate outputs.")

    # print(f"First mismatched intermediate output: {mismatched[0]}")
