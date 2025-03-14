"""
Validate SIMS vs. ONNX generate concordant output

This script manually digs into various parts of the SIMS computation and compares
to onnx on the python runtime and output from the website saved as csv. Its
primarily a tool to explore the compute graph when you're doing surgery to expose outputs
and/or diagnose numerical differences. There are direct links to lines in Tabnet and SIMS throughout
to assist understanding which part we're trying to validate. Now that the model is concordant this step
by step seems overkill - just compare end to end. But when trying to identify a problem such as the
softmax numerical difference its helpful to see where things go wrong...

SIM().network is a Tabnet:
https://github.com/braingeneers/SIMS/blob/e648db22a640da3dba333e86154ace1599dba267/scsims/model.py#L101

https://github.com/dreamquark-ai/tabnet/blob/2m0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L508

TabNet.forward:
https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L614

calls TabNetNoEmbeddings(EmbeddingsGenerator(x))

EmbeddingsGenerator:
https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L809

BUT skip_embeddings = True so its just TabNetNoEmbeddings(x)

TabNetNoEmbeddings:
https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L399

So...its really at its core sims.model.network.tabnet which is:
TabNetNoEmbeddings.forward
and
TabNetNoEmbeddings.forward_masks

https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L490
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.onnx
from torch.onnx.verification import verify
import anndata
import onnx
import onnxruntime as ort
import sclblonnx as so
from scsims import SIMS


# Directory for temporary output files such as .onnx for parts of the model
TEMP_OUTPUT = "data/validation"


def print_diffs(msg, a, b, decimals=5):
    diff = np.round(np.abs(a - b).flatten(), decimals)
    print(
        "{} differ by {} values out of {} to {} decimals".format(
            msg,
            np.count_nonzero(diff),
            len(diff),
            decimals,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to a SIMS checkpoint")
    parser.add_argument(
        "onnx", type=str, help="Path production ONNX model from checkpoint"
    )
    parser.add_argument("sample", type=str, help="Path to a sample h5ad file")
    parser.add_argument("--decimals", type=int, default=5, help="# decimals to compare")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print(f"Validating to {args.decimals} decimals")
    def close(a, b, decimals=args.decimals):
        """ Returns True if a and b are close to each other to the number of decimals specified """
        return np.allclose(a, b, rtol=10**-decimals, atol=10**-decimals)

    # Load the checkpoint
    print("Loading checkpoint...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
    )

    # Set the seed so this is repeatable
    _ = torch.manual_seed(42)

    # Generate a random batch of data and a normalized version
    # SIMS data loader normalizes data as its loaded and lpnorm is added to the exported ONNX graph
    # When comparing raw components of SIMS in python to node outputs in ONNX we need to provide
    # the un-normalized to the ONNX graph and the normalized to the raw python code. This is
    # the most common source of error in these tests so think through which makes sense when adding
    # and modifying...YMMV.
    x_unnormalized = torch.randn(args.batch_size, len(sims.model.genes))
    x_norm = torch.nn.functional.normalize(x_unnormalized, p=2, dim=1)

    """
    Encoders
    https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L490
    """
    # Export just the trained encoder from SIMS
    model = sims.model.network.tabnet.encoder
    _ = model.eval()
    _ = torch.onnx.export(
        model,
        torch.zeros(1, len(sims.model.genes)),
        f"{TEMP_OUTPUT}/encoder.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits"],
        export_params=True,
    )

    # Create an ONNX runtime session for just the encoder and the entire model (including lpnorm!)
    encoder_onnx_session = ort.InferenceSession(f"{TEMP_OUTPUT}/encoder.onnx")
    sims_onnx_session = ort.InferenceSession(args.onnx)

    # Compare overall encoder outputs as well as steps
    for i in range(args.batch_size):
        steps_output_onnx = encoder_onnx_session.run(None, {"input": x_norm[i : i + 1].detach().numpy()})
        steps_output_py, _ = sims.model.network.tabnet.encoder.forward(x_norm[i : i + 1])
        for j, (py_step, onnx_step) in enumerate(zip(steps_output_py, steps_output_onnx)):
            print(
                f"Encoder Sample {i} Step {j} Is Close:",
                close(py_step.detach().numpy(), onnx_step),
            )

        encoder_py = torch.sum(torch.stack(steps_output_py, dim=0), dim=0)
        encoder_onnx = sims_onnx_session.run(["encoding"], {"input": x_unnormalized[i : i + 1].detach().numpy()})
        print(
            f"Encoder Sample {i} Is Close:",
            close(encoder_py.detach().numpy(), encoder_onnx[0]),
        )

        

    """
    FinalMappings
    https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L501
    """

    model = sims.model.network.tabnet.final_mapping
    _ = model.eval()

    onnx_program = torch.onnx.export(
        model,
        steps_output_py[0],
        f"{TEMP_OUTPUT}/mappings.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession(f"{TEMP_OUTPUT}/mappings.onnx")
    res_py = torch.sum(torch.stack(steps_output_py, dim=0), dim=0)
    res_onnx = np.sum(np.stack(steps_output_onnx[0:3], axis=0), axis=0)
    print(
        "Mappings Res Is Close:",
        close(res_py.detach().numpy(), res_onnx),
    )
    out_py = sims.model.network.tabnet.final_mapping(res_py)
    out_onnx = session.run(None, {"input": res_onnx})
    print_diffs("mappings", out_py.detach().numpy(), out_onnx, args.decimals)
    print(
        "Mappings Is Close:",
        close(out_py.detach().numpy(), out_onnx),
    )

    """
    Logits from full core TabNet
    """

    model = sims.model.network
    _ = model.eval()
    onnx_program = torch.onnx.export(
        model,
        torch.randn(1, len(sims.model.genes)),
        f"{TEMP_OUTPUT}/logits.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession(f"{TEMP_OUTPUT}/logits.onnx")
    for i in range(args.batch_size):
        logits_onnx = session.run(None, {"input": x_norm[i : i + 1].detach().numpy()})
        logits_py = sims.model.network.forward(x_norm[i : i + 1])
        print(
            f"Logits {i} Is Close:",
            close(logits_py[0][0].detach().numpy(), logits_onnx[0][0]),
        )

    """
    SIMS runs torch.nn.functional.normalize on the input data before passing it to the model.
    """
    onnx_model = onnx.load(args.onnx)
    path = "lpnorm"
    _ = so.add_output(onnx_model.graph, path, "FLOAT", x_unnormalized.shape)
    p_norm = so.run(
        onnx_model.graph, inputs={"input": x_unnormalized.detach().numpy()}, outputs=[path]
    )[0]
    print(
        "LpNorm Is Close:",
        close(p_norm, x_norm),
    )

    """
    Full probability predictions end to end on a real sample
    """
    num_samples = 100
    sims_predictions = sims.predict(args.sample, rows=list(range(num_samples)))
    # The onnx model has lpnorm normalization built in so we need non-normalized
    batch = next(
        enumerate(sims.model._parse_data(args.sample, num_samples, normalize=False))
    )
    session = ort.InferenceSession(args.onnx)
    for i in range(num_samples):
        onnx_predictions = session.run(
            ["probs"], {"input": batch[1][i : i + 1].to(torch.float32).detach().numpy()}
        )
        print(
            f"Prediction {i} Is Close:",
            close(onnx_predictions[0][0], sims_predictions.values[i][3:6].astype(np.float32)),
        )

    """
    Compare downloaded csv from web app to predictions for this sample from SIMS
    """
    # csv downloaded from current web app
    onnx_web_predictions = pd.read_csv("validation/predictions.csv")

    # predictions on sample.h5ad from a clean production/published SIMS install
    sims_predictions = pd.read_csv("validation/gold-sample-predictions.csv")
    print(
        "Prediction 0 Is Close:",
        close(onnx_web_predictions.prob_0.values, sims_predictions.prob_0.values),
    )
    print(
        "Prediction 1 Is Close:",
        close(onnx_web_predictions.prob_1.values, sims_predictions.prob_1.values),
    )
    print(
        "Prediction 2 Is Close:",
        close(onnx_web_predictions.prob_2.values, sims_predictions.prob_2.values),
    )

    """
    Masks
    https://github.com/braingeneers/SIMS/blob/e648db22a640da3dba333e86154ace1599dba267/scsims/model.py#L268

    Calls self.network.forward_masks(X)
    """
    M_explain, masks = sims.model.network.forward_masks(x_norm)

    model = sims.model.network
    _ = model.eval()
    onnx_model = torch.onnx.export(
        model,
        torch.zeros(1, len(sims.model.genes)),
        f"{TEMP_OUTPUT}/masks.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession(f"{TEMP_OUTPUT}/masks.onnx")

    onnx_model = onnx.load(f"{TEMP_OUTPUT}/masks.onnx")

    paths = [
        "/tabnet/encoder/att_transformers.0/selector/Clip_output_0",
        "/tabnet/encoder/att_transformers.1/selector/Clip_output_0",
        "/tabnet/encoder/att_transformers.2/selector/Clip_output_0",
    ]
    for path in paths:
        shape_info = onnx.shape_inference.infer_shapes(onnx_model)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name == path:
                # print(idx, node)
                break
        assert node.name == path
        onnx_model.graph.output.extend([node])
    onnx.save(onnx_model, f"{TEMP_OUTPUT}/masks.onnx")
    session = ort.InferenceSession(f"{TEMP_OUTPUT}/masks.onnx")

    onnx_masks = session.run(None, {"input": x_norm[0 : 1].detach().numpy()})[2:]

    for i, path in enumerate(paths):
        print_diffs(
            f"mask {i}",
            masks[i][0].detach().numpy(),
            onnx_masks[i][0],
            args.decimals,
        )
        print(
            "# non zero values in the mask:",
            np.count_nonzero(masks[i][0].detach().numpy()),
        )

    """
    Explain
    https://github.com/braingeneers/SIMS/blob/e648db22a640da3dba333e86154ace1599dba267/scsims/model.py#L268
    """
    M_explain, masks = sims.explain(args.sample, batch_size=args.batch_size, rows=[0])
    np.count_nonzero(M_explain[0])

    np.count_nonzero(onnx_masks[0])
    np.count_nonzero(onnx_masks[1])
    np.count_nonzero(onnx_masks[2])

    # onnx_explain = np.sum(onnx_masks, axis=0)
    onnx_explain = onnx_masks[0][0] * onnx_masks[1][0] * onnx_masks[2][0]
    print(
        "#s non zero values in onnx_explain",
        np.count_nonzero(onnx_explain[0]),
    )
    print_diffs("explain", M_explain[0], onnx_explain[0], 2)
