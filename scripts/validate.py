"""
Compare SIMS vs. ONNX

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
import anndata
import onnx
from onnxruntime import InferenceSession
from scsims import SIMS
import sclblonnx as so


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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    if args.batch_size != 1:
        print(
            "\033[92m WARNING: Batch size > 1 generates output inconsistent with SIMS/PyTorch \033[0m"
        )

    print("Loading SIMS model...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()  # Turns off training mode

    # The onnx model runs lpnorm so we need non-normalized to compare end to end
    batch_un_normalized = next(
        enumerate(
            sims.model._parse_data(
                args.sample, batch_size=args.batch_size, normalize=False
            )
        )
    )[1].to(torch.float32)
    batch = next(
        enumerate(
            sims.model._parse_data(
                args.sample, batch_size=args.batch_size, normalize=True
            )
        )
    )[1].to(torch.float32)
    # Verify understanding of what the data loader is doing
    assert np.allclose(batch[0], torch.nn.functional.normalize(batch_un_normalized[0], dim=0))

    # Normalized random input 
    _ = torch.manual_seed(42)
    x_un_normalized = torch.randn(args.batch_size, len(sims.model.genes))
    x = torch.nn.functional.normalize(x_un_normalized, dim=0)

    """
    Encoders
    https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L490
    """
    encoder_path = "data/validation/encoder.onnx"
    sims.model.network.tabnet.encoder.eval()
    torch.onnx.export(
        sims.model.network.tabnet.encoder,
        torch.zeros(1, len(sims.model.genes)),
        encoder_path,
        opset_version=12,  # 12 works in web runtime, later doesn't
        # do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        # dynamic_axes={"input": {0: "batch_size"}},
    )
    session = InferenceSession(encoder_path)

    steps_output_onnx = session.run(None, {"input": x.detach().numpy()})
    steps_output_py, _ = sims.model.network.tabnet.encoder.forward(x)
    print_diffs(
        "first encoder",
        steps_output_py[0].detach().numpy(),
        steps_output_onnx[0],
        args.decimals,
    )
    print_diffs(
        "first encoder",
        steps_output_py[1].detach().numpy(),
        steps_output_onnx[1],
        args.decimals,
    )
    print_diffs(
        "first encoder",
        steps_output_py[2].detach().numpy(),
        steps_output_onnx[2],
        args.decimals,
    )

    # _ = torch.manual_seed(42)
    x_un_normalized = torch.randn(args.batch_size, len(sims.model.genes))
    x = torch.nn.functional.normalize(x_un_normalized, dim=0)
    for i in range(args.batch_size):
        steps_output_onnx = session.run(None, {"input": x[i : i + 1].detach().numpy()})
        steps_output_py, _ = sims.model.network.tabnet.encoder.forward(x[i : i + 1])
        for j in range(len(steps_output_py)):
            print_diffs(
                f"sample {i} encoder {j}",
                steps_output_py[j].detach().numpy(),
                steps_output_onnx[j],
                args.decimals,
            )


    for i in range(args.batch_size):
        steps_output_onnx = session.run(None, {"input": batch[i : i + 1].detach().numpy()})
        steps_output_py, _ = sims.model.network.tabnet.encoder.forward(batch[i : i + 1])
        for j in range(len(steps_output_py)):
            print_diffs(
                f"sample {i} encoder {j}",
                steps_output_py[j].detach().numpy(),
                steps_output_onnx[j],
                args.decimals,
            )

    """
    FinalMappings 
    https://github.com/dreamquark-ai/tabnet/blob/2c0c4ebd2bb1cb639ea94ab4b11823bc49265588/pytorch_tabnet/tab_network.py#L501
    """

    mapper_path = "data/validation/mappings.onnx"
    _ = sims.model.network.tabnet.final_mapping.eval()
    torch.onnx.export(
        sims.model.network.tabnet.final_mapping,
        torch.zeros(1, steps_output_py[0][0].shape[0]),
        mapper_path,
        opset_version=12,  # 12 works in web runtime, later doesn't
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
    session = InferenceSession(mapper_path)
    res_py = torch.sum(torch.stack(steps_output_py, dim=0), dim=0)
    res_onnx = np.sum(np.stack(steps_output_onnx[0:3], axis=0), axis=0)
    print_diffs("res", res_py.detach().numpy(), res_onnx[0], args.decimals)
    out_py = sims.model.network.tabnet.final_mapping(res_py)
    out_onnx = session.run(None, {"input": res_onnx})
    print_diffs("mappings", out_py.detach().numpy(), out_onnx, args.decimals)

    """
    Logits from full core TabNet
    """
    logits_path = "data/validation/logits.onnx"
    sims.model.network.eval()
    torch.onnx.export(
        sims.model.network.tabnet,
        torch.zeros(1, len(sims.model.genes)),
        logits_path,
        opset_version=12,  # 12 works in web runtime, later doesn't
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
    session = InferenceSession(logits_path)
    logits_onnx = session.run(None, {"input": x.detach().numpy()})
    logits_py = sims.model.network.forward(x)
    for i in range(args.batch_size):
        print_diffs(
            f"logits sample {i}",
            logits_py[0][i].detach().numpy(),
            logits_onnx[0][i],
            args.decimals,
        )

    # Compare the full sample's logits to onnx
    for i in range(batch.shape[0]):
        logits_py = sims.model.network.forward(batch[i : i + 1].to(torch.float32))
        logits_onnx = session.run(
            None, {"input": batch[i : i + 1].to(torch.float32).detach().numpy()}
        )
        print_diffs(
            f"logits sample {i}", logits_py[0][0].detach().numpy(), logits_onnx[0][0], 2
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
    print_diffs("norms", p_norm, x_norm, args.decimals)

    """
    Full probability predictions end to end
    """
    num_samples = 100
    sims_predictions = sims.predict(args.sample, rows=list(range(num_samples)))

    # The onnx model has lpnorm normalization built in so we need non-normalized
    batch = next(
        enumerate(sims.model._parse_data(args.sample, num_samples, normalize=False))
    )
    session = InferenceSession(args.onnx)

    # Do one at a time - see above
    for i in range(num_samples):
        onnx_predictions = session.run(
            ["probs"], {"input": batch[1][i : i + 1].to(torch.float32).detach().numpy()}
        )
        # print(onnx_predictions[0][0])
        # print(sims_predictions.values[i][3:6])
        print_diffs(
            f"sample {i} probs",
            onnx_predictions[0][0],
            sims_predictions.values[i][3:6].astype(np.float32),
            3,
        )

    """
    Compare downloaded csv from web app to predictions for this sample from SIMS
    """
    onnx_web_predictions = pd.read_csv("data/validation/predictions-no-batch.csv")
    sims_predictions = sims.predict(args.sample)

    print_diffs(
        "onnx web vs. sims",
        onnx_web_predictions.prob_0.values,
        sims_predictions.prob_0.values,
        3,
    )
    print_diffs(
        "onnx web vs. sims",
        onnx_web_predictions.prob_1.values,
        sims_predictions.prob_1.values,
        3,
    )

    """
    Masks
    https://github.com/braingeneers/SIMS/blob/e648db22a640da3dba333e86154ace1599dba267/scsims/model.py#L268

    Calls self.network.forward_masks(X)
    """
    M_explain, masks = sims.model.network.forward_masks(x)

    masks_path = "data/validation/masks.onnx"
    sims.model.network.eval()
    torch.onnx.export(
        sims.model.network.tabnet,
        torch.zeros(1, len(sims.model.genes)),
        masks_path,
        opset_version=12,  # 12 works in web runtime, later doesn't
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        # dynamic_axes={"input": {0: "batch_size"}},
    )
    onnx_model = onnx.load(masks_path)

    paths = [
        "/encoder/att_transformers.0/selector/Clip_output_0",
        "/encoder/att_transformers.1/selector/Clip_output_0",
        "/encoder/att_transformers.2/selector/Clip_output_0",
    ]
    for path in paths:
        shape_info = onnx.shape_inference.infer_shapes(onnx_model)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name == path:
                # print(idx, node)
                break
        assert node.name == path
        onnx_model.graph.output.extend([node])

    onnx_masks = so.run(
        onnx_model.graph, inputs={"input": x.detach().numpy()}, outputs=paths
    )
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
    # M_explain, masks = sims.explain(args.sample, batch_size=args.batch_size, rows=[0])

    # np.count_nonzero(M_explain[0])

    # onnx_masks = so.run(
    #     onnx_model.graph,
    #     inputs={"input": batch[1][0:1].to(torch.float32).detach().numpy()},
    #     outputs=paths,
    # )
    # np.count_nonzero(onnx_masks[0])
    # np.count_nonzero(onnx_masks[1])
    # np.count_nonzero(onnx_masks[2])

    # # onnx_explain = np.sum(onnx_masks, axis=0)
    # onnx_explain = onnx_masks[0][0] * onnx_masks[1][0] * onnx_masks[2][0]
    # print(
    #     "# non zero values in onnx_explain",
    #     np.count_nonzero(onnx_explain[0]),
    # )
    # print_diffs("explain", M_explain[0].detach().numpy(), onnx_explain[0], 2)

    """
    WebRuntime vs. Python
    """
    # from selenium import webdriver
    # import chromedriver_binary  # Adds chromedriver binary to path

    # driver = webdriver.Chrome()
    # driver.get("http://localhost:5173")
    # assert "Python" in driver.title
