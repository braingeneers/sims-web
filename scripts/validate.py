"""
Validate SIMS vs. ONNX generate concordant output

This script manually digs into various parts of the SIMS computation and compares
to onnx on the python runtime and output from the website saved as csv. Its
primarily a tool to explore the compute graph when you're doing surgery to expose outputs
and/or diagnose numerical differences.

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

    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
    )

    _ = torch.manual_seed(42)
    x = torch.nn.functional.normalize(
        torch.randn(args.batch_size, len(sims.model.genes)), dim=0
    )

    """
    Encoders
    https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L490
    """

    model = sims.model.network.tabnet.encoder
    _ = model.eval()

    onnx_pgm = torch.onnx.export(
        model,
        torch.zeros(1, len(sims.model.genes)),
        "data/validation/encoder.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits"],
        export_params=True,
        # external_data=False,
        # dynamo=True,
        # optimize=True,
        # verbose=True,
        # verify=True,
        # opset_version=12,
        # dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load("data/validation/encoder.onnx")

    # onnx_program = torch.onnx.export(
    #     model,
    #     torch.zeros(1, len(sims.model.genes)),
    #     dynamo=True,
    # )
    # onnx_program.optimize()
    # onnx_program.save("data/validation/encoder.onnx")

    # verification_options = torch.onnx.verification.VerificationOptions(
    #     flatten=True,
    #     check_shape=False,
    #     rtol=1.3e-3,
    #     atol=1e-4,
    # )
    # verify(model, x[0:1], options=verification_options)

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # session = ort.InferenceSession("data/validation/encoder.onnx", sess_options)
    session = ort.InferenceSession("data/validation/encoder.onnx")

    for i in range(4):
        steps_output_onnx = session.run(None, {"input": x[i : i + 1].detach().numpy()})
        steps_output_py, _ = sims.model.network.tabnet.encoder.forward(x[i : i + 1])
        for j in range(len(steps_output_py)):
            print(
                f"Encodings {i}/{j} Is Close:",
                np.allclose(
                    steps_output_py[j].detach().numpy(),
                    steps_output_onnx[j],
                    rtol=1e-3,
                    atol=1e-4,
                ),
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
        "data/validation/mappings.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession("data/validation/mappings.onnx")
    res_py = torch.sum(torch.stack(steps_output_py, dim=0), dim=0)
    res_onnx = np.sum(np.stack(steps_output_onnx[0:3], axis=0), axis=0)
    print(
        "Mappings Res Is Close:",
        np.allclose(
            res_py.detach().numpy(),
            res_onnx,
            rtol=1e-3,
            atol=1e-4,
        ),
    )
    out_py = sims.model.network.tabnet.final_mapping(res_py)
    out_onnx = session.run(None, {"input": res_onnx})
    print_diffs("mappings", out_py.detach().numpy(), out_onnx, args.decimals)
    print(
        "Mappings Is Close:",
        np.allclose(
            out_py.detach().numpy(),
            out_onnx,
            rtol=1e-3,
            atol=1e-4,
        ),
    )

    """
    Logits from full core TabNet
    """

    model = sims.model.network
    _ = model.eval()
    onnx_program = torch.onnx.export(
        model,
        torch.randn(1, len(sims.model.genes)),
        "data/validation/logits.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession("data/validation/logits.onnx")
    for i in range(args.batch_size):
        logits_onnx = session.run(None, {"input": x[i : i + 1].detach().numpy()})
        logits_py = sims.model.network.forward(x[i : i + 1])
        print(
            f"Logits {i} Is Close:",
            np.allclose(
                logits_py[0][0].detach().numpy(),
                logits_onnx[0][0],
                rtol=1e-3,
                atol=1e-4,
            ),
        )

    """
    SIMS runs torch.nn.functional.normalize on the input data before passing it to the model.
    """
    onnx_model = onnx.load(args.onnx)
    path = "lpnorm"
    _ = so.add_output(onnx_model.graph, path, "FLOAT", x.shape)
    p_norm = so.run(
        onnx_model.graph, inputs={"input": x.detach().numpy()}, outputs=[path]
    )[0]
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1).detach().numpy()
    print(
        "LpNorm Is Close:",
        np.allclose(
            p_norm,
            x_norm,
            rtol=1e-3,
            atol=1e-4,
        ),
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
            np.allclose(
                onnx_predictions[0][0],
                sims_predictions.values[i][3:6].astype(np.float32),
                rtol=1e-3,
                atol=1e-4,
            ),
        )

    """
    Compare downloaded csv from web app to predictions for this sample from SIMS
    """
    onnx_web_predictions = pd.read_csv("data/validation/predictions.csv")
    sims_predictions = pd.read_csv("data/validation/predictions.gold.csv")
    print(
        "Prediction 0 Is Close:",
        np.allclose(
            onnx_web_predictions.prob_0.values,
            sims_predictions.prob_0.values,
            rtol=1e-3,
            atol=1e-4,
        ),
    )
    print(
        "Prediction 1 Is Close:",
        np.allclose(
            onnx_web_predictions.prob_1.values,
            sims_predictions.prob_1.values,
            rtol=1e-3,
            atol=1e-4,
        ),
    )
    print(
        "Prediction 2 Is Close:",
        np.allclose(
            onnx_web_predictions.prob_2.values,
            sims_predictions.prob_2.values,
            rtol=1e-3,
            atol=1e-4,
        ),
    )

    """
    Masks
    https://github.com/braingeneers/SIMS/blob/e648db22a640da3dba333e86154ace1599dba267/scsims/model.py#L268

    Calls self.network.forward_masks(X)
    """
    M_explain, masks = sims.model.network.forward_masks(x)

    model = sims.model.network
    _ = model.eval()
    onnx_model = torch.onnx.export(
        model,
        torch.zeros(1, len(sims.model.genes)),
        "data/validation/masks.onnx",
        input_names=["input"],
    )
    session = ort.InferenceSession("data/validation/masks.onnx")

    onnx_model = onnx.load("data/validation/masks.onnx")

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
    onnx.save(onnx_model, "data/validation/masks.onnx")
    session = ort.InferenceSession("data/validation/masks.onnx")

    onnx_masks = session.run(None, {"input": x[i : i + 1].detach().numpy()})[2:]

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
        "# non zero values in onnx_explain",
        np.count_nonzero(onnx_explain[0]),
    )
    print_diffs("explain", M_explain[0], onnx_explain[0], 2)
