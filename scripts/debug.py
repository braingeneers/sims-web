"""
Scratch to debug differences between pytorch and onnx compute

Minimum install details to get a local SIMS running same as published
but with editable SIMS and tabnet repos:

Works:
rm -rf venv
python3.10 -m venv venv
pip install git+https://github.com/braingeneers/SIMS.git
pip install onnx==1.17.0 onnxruntime==1.20.1 onnxscript==0.2.0 onnxsim==0.4.36 sclblonnx==0.3.0
rm data/validation/*
python scripts/debug.py

Works:
rm -rf venv
python3.10 -m venv venv
cd ../SIMS && git checkout e648db22a640da3dba333e86154ace1599dba267
pip install -e ../SIMS
pip install onnx==1.17.0 onnxruntime==1.20.1 onnxscript==0.2.0 onnxsim==0.4.36 sclblonnx==0.3.0
rm data/validation/*
python scripts/debug.py

Using rcurrie onnx branches of SIMS and tabnet:
rm -rf venv
python3.10 -m venv venv
pip install -e ../SIMS
pip install -e ../tabnet
pip install onnx==1.17.0 onnxruntime==1.20.1 onnxscript==0.2.0 onnxsim==0.4.36 sclblonnx==0.3.0
rm data/validation/*
python scripts/debug.py
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import pytorch_tabnet.sparsemax
import scsims
import sclblonnx as so

if __name__ == "__main__":

    double = False

    sims = scsims.SIMS(
        weights_path="checkpoints/default.ckpt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model = sims.model.network.double() if double else sims.model.network
    _ = model.eval()

    torch.onnx.export(
        model,
        (
            torch.zeros(1, len(sims.model.genes)).double()
            if double
            else torch.zeros(1, len(sims.model.genes))
        ),
        "data/validation/default.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        # verbose=True,
        # dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        "data/validation/default.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    num_samples = 100
    _ = torch.manual_seed(42)
    x = torch.nn.functional.normalize(
        torch.randn(num_samples, len(sims.model.genes)), dim=1
    )
    if double:
        x = x.double()

    # Compare logit outputs
    print("logits pytorch vs. onnx # decimal places of agreement for each sample:")
    for i in range(num_samples):
        a = session.run(None, {"input": x[i : i + 1].cpu().numpy()})[0][0]
        b = model.forward(x[i : i + 1])[0][0].detach().numpy()
        overall_min = np.min(np.floor(-np.log10(np.abs(a - b) + 1e-16)).astype(int))
        print(f"{overall_min} ", end="")
        # assert np.allclose(a, b, rtol=1e-4, atol=1e-5)
        # print("")
    print("")

    a = model.forward(x)[0].detach().numpy()
    b = pd.read_csv("validation/gold-x-logits.csv").to_numpy()[0:100, 1:]
    overall_min = np.min(np.floor(-np.log10(np.abs(a - b) + 1e-16)).astype(int))
    print(f"logits gold vs. current: {overall_min}")
