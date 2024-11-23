import argparse
import sys
import torch
import torch.onnx
import onnx

import sclblonnx as so

# Assumes you have the SIMS repo as a peer to this one...
sys.path.insert(0, "../SIMS")
from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a SIMS model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Load the checkpoint
    print("Loading model...")
    sims = SIMS(weights_path=args.checkpoint, map_location=torch.device("cpu"))
    model_name = args.checkpoint.split("/")[-1].split(".")[0]
    model_path = "/".join(args.checkpoint.split("/")[:-1])
    num_genes = len(sims.model.genes)
    print(f"Loaded model {model_name} with {num_genes} genes")

    # Export model to ONNX file
    # wrt batch size https://github.com/microsoft/onnxruntime/issues/19452#issuecomment-1947799229
    batch_size = 1
    sims.model.to_onnx(
        f"{model_path}/{model_name}.onnx",
        torch.zeros(batch_size, num_genes),
        export_params=True,
    )
    print(f"Exported model to {model_path}/{model_name}.onnx")

    model = onnx.load(f"{model_path}/{model_name}.onnx")
    onnx.checker.check_model(model)
    g = model.graph
    print("Original outputs")
    so.list_outputs(g)

    # g = so.rename_output(g, "826", "logits")
    # so.list_outputs(g)

    # Modify the model to add an ArgMax and Softmax output
    n1 = so.node("ArgMax", inputs=["826"], outputs=["argmax"], keepdims=0, axis=1)
    g = so.add_node(g, n1)
    g = so.add_output(g, "argmax", "INT64", [1])

    n2 = so.node("Softmax", inputs=["826"], outputs=["softmax"], axis=1)
    g = so.add_node(g, n2)
    g = so.add_output(g, "softmax", "FLOAT", [1, 8])

    print("New outputs")
    so.list_outputs(g)

    # Save the modified ONNX model
    onnx.save(model, f"{model_path}/{model_name}.onnx")
    print(f"Saved modified model to {model_path}/{model_name}.onnx")

    # Write out the list of genes corresponding to the models input
    with open(f"{model_path}/{model_name}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    print(f"Wrote out gene list to {model_path}/{model_name}.genes")

    # Write out the class labels for the model
    with open(f"{model_path}/{model_name}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out classes to {model_path}/{model_name}.classes")
