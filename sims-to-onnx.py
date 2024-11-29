import argparse
import sys
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime

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
    model_input_size = sims.model.input_dim
    assert num_genes == model_input_size
    print(f"Loaded {model_name} with {num_genes} genes")

    # Write out the list of genes corresponding to the models input
    with open(f"{model_path}/{model_name}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    print(f"Wrote out gene list to {model_path}/{model_name}.genes")

    # Write out the class labels for the model
    with open(f"{model_path}/{model_name}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out classes to {model_path}/{model_name}.classes")

    test_tensor = torch.zeros(1, model_input_size)
    for i in range(0, model_input_size, 8):
        test_tensor[0, i] = 0.5
    # For log1p testing
    test_tensor[0, 0] = 0.0
    test_tensor[0, 1] = 1.0
    test_tensor[0, 2] = 1.5

    # Export model to ONNX file
    # wrt batch size https://github.com/microsoft/onnxruntime/issues/19452#issuecomment-1947799229
    batch_size = 1
    sims.model.to_onnx(
        f"{model_path}/{model_name}.onnx",
        torch.zeros(batch_size, num_genes),
        export_params=True,
    )
    print(f"Exported model to {model_path}/{model_name}.onnx")

    # Run the model on the test tensor
    sims.model.eval()  # This is necessary to avoid dropout
    results = sims.model(test_tensor.float())

    session = onnxruntime.InferenceSession(f"{model_path}/{model_name}.onnx")
    outputs = session.run(None, {"input.1": test_tensor.numpy()})

    print("sims.model vs. onnxruntime output:")
    print(results[0][0].detach().numpy())
    print(outputs[0][0])

    # Load the model back in as onnx for checking and editing
    model = onnx.load(f"{model_path}/{model_name}.onnx")
    onnx.checker.check_model(model)
    g = model.graph
    print("Original inputs")
    so.list_inputs(g)
    print("Original outputs")
    so.list_outputs(g)

    # Doesn't seem to work...
    # g = so.rename_output(g, "826", "logits")
    # so.list_outputs(g)

    # Modify the model to add an ArgMax and Softmax output
    n = so.node("ArgMax", inputs=["826"], outputs=["argmax"], keepdims=0, axis=1)
    g = so.add_node(g, n)
    g = so.add_output(g, "argmax", "INT64", [1])

    n = so.node("Softmax", inputs=["826"], outputs=["softmax"], axis=1)
    g = so.add_node(g, n)
    g = so.add_output(
        g, "softmax", "FLOAT", [1, len(sims.model.label_encoder.classes_)]
    )

    print("New outputs")
    so.list_outputs(g)

    # Save the modified ONNX model
    onnx.save(model, f"{model_path}/{model_name}.onnx")
    print(f"Saved modified model to {model_path}/{model_name}.onnx")

    """
    Save a separate log1p graph to run when we are processing raw h5 files
    """
    # debug
    # model_input_size = 9

    # Create a ln(1 + x) normalization graph
    g = so.empty_graph("log1p")
    # Can use broadcast here, but we'll just add a constant
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    # g = so.add_constant(g, "one", np.ones((1, model_input_size)), "FLOAT")
    g = so.add_constant(g, "one", np.ones((1, 1)), "FLOAT")
    g = so.add_input(g, "raw", "FLOAT", [1, model_input_size])
    n = so.node("Add", inputs=["raw", "one"], outputs=["1p"])
    g = so.add_node(g, n)
    n = so.node("Log", inputs=["1p"], outputs=["log1p"])
    g = so.add_node(g, n)
    g = so.add_output(g, "log1p", "FLOAT", [1, model_input_size])
    so.check(g)
    print("log1p inputs")
    so.list_inputs(g)
    print("log1p outputs")
    so.list_inputs(g)
    so.list_outputs(g)

    # Test the log1p graph
    result = so.run(g, inputs={"raw": test_tensor.numpy()}, outputs=["log1p"])
    print("log1p graph test")
    print(result)
    print("Should be:", np.log1p([0.0, 1.0, 1.5]))

    so.graph_to_file(g, f"{model_path}/{model_name}.log1p.onnx")
    print(f"Saved log1p graph to {model_path}/{model_name}.log1p.onnx")

    # # Create a combined log1p + model graph
    # g1 = so.graph_from_file(f"{model_path}/{model_name}.log1p.onnx")
    # g2 = so.graph_from_file(f"{model_path}/{model_name}.onnx")
    # g12 = so.merge(sg1=g1, sg2=g2, io_match=[("log1p", "input.1")], complete=False)
    # so.check(g1, _sclbl_check=False)
    # so.check(g2, _sclbl_check=False)
    # so.check(g12, _sclbl_check=False)

    # so.list_inputs(g12)
    # so.list_outputs(g12)

    # so.graph_to_file(g12, f"{model_path}/{model_name}.combined.onnx")

    # session = onnxruntime.InferenceSession(f"{model_path}/{model_name}.combined.onnx")
    # outputs = session.run(None, {"raw": test_tensor.numpy()})

    # model = onnx.load(f"{model_path}/{model_name}.combined.onnx")

    # result = so.run(g12, inputs={"raw": test_tensor.numpy()}, outputs=["826"])
    # print("log1p graph test")
    # print(result)
    # print("Should be:", np.log1p([0.0, 1.0, 1.5]))
