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
    parser.add_argument("--decimals", type=int, default=5, help="# decimals to compare")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    print("Loading SIMS model...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()  # Turns off training mode

    # Normalized random input for the encoder to better approximate real data
    _ = torch.manual_seed(42)
    x = torch.randn(args.batch_size, len(sims.model.genes))
    x = torch.nn.functional.normalize(x)

    # # This generates comparable encoding outputs with real data from an h5ad file
    # batch = next(
    #     enumerate(sims.model._parse_data("public/sample.h5ad", batch_size=args.batch_size))
    # )
    # x = batch[1].to(torch.float32)

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
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        dynamic_axes={"input": {0: "batch_size"}},
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
        steps_output_py[0].detach().numpy(),
        steps_output_onnx[0],
        args.decimals,
    )
    print_diffs(
        "first encoder",
        steps_output_py[0].detach().numpy(),
        steps_output_onnx[0],
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
        sims.model.network,
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


# def validate_masks(sm, pm, cm, x):
#     M_explain, masks = sm.network.forward_masks(x)
#     path = "/network/tabnet/encoder/att_transformers.0/selector/Clip_output_0"
#     shape_info = onnx.shape_inference.infer_shapes(pm)
#     for idx, node in enumerate(shape_info.graph.value_info):
#         if node.name == candidate:
#             print(idx, node)
#             break
#     assert node.name == candidate
#     model.graph.output.extend([node])


# # Export the model
# torch.onnx.export(
#     model,  # model being run
#     dummy_input,  # model input (or a tuple for multiple inputs)
#     "model.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=13,  # the ONNX version to export the model to
#     do_constant_folding=True,  # whether to execute constant folding for optimization
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={
#         "input": {0: "batch_size"},  # variable length axes
#         "output": {0: "batch_size"},
#     },
# )

# validate_preprocessing(sm, pm, cm, x)

# # Load count cells from the sample
# adata = anndata.read(args.sample)[0 : args.count]

# # The onnx model has lpnorm normalization built in so we need non-normalized
# # and normalized batches from the sample to compare to the raw onnx model
# batch_size = adata.shape[0]
# batch = next(enumerate(sm._parse_data(args.sample, batch_size, normalize=False)))
# x = batch[1].to(torch.float32)
# batch_normalized = next(
#     enumerate(sm._parse_data(args.sample, batch_size, normalize=True))
# )
# x_normalized = batch_normalized[1].to(torch.float32)

# # Compare sims vs. production onnx model predictions
# sp = sm.predict(adata)
# sp_probs = sp.prob_0.values

# output = so.run(pm.graph, inputs={"input": x.detach().numpy()}, outputs=["probs"])
# pm_probs = [p[0] for p in output[0]]

# print(
#     "sims python vs. onnx production {} prob_0 differ by more then {} decimals out of {}".format(
#         np.count_nonzero(
#             np.round(np.abs(sp_probs - pm_probs), decimals=args.decimals)
#         ),
#         args.decimals,
#         len(pm_probs),
#     )
# )

# opset_version = cm.opset_import[0].version if len(cm.opset_import) > 0 else None
# print("All ONNX models using opset version ", opset_version)

# # At this point we have sm, cm, and pm models loaded

# #
# # Compare raw logits
# #
# sm_logits = sm(x_normalized)[0].detach().numpy()
# cm_logits = so.run(
#     cm.graph,
#     inputs={"input": x_normalized.detach().numpy()},
#     outputs=["logits"],
# )[0]
# pm_logits = so.run(
#     pm.graph,
#     inputs={"input": x.detach().numpy()},
#     outputs=["logits"],
# )[0]
# print(
#     "cm vs. pm {} logits differ by more then {} decimals out of {}".format(
#         np.count_nonzero(
#             np.round(np.abs(cm_logits - pm_logits), decimals=args.decimals)
#         ),
#         args.decimals,
#         pm_logits.shape[0] * pm_logits.shape[1],
#     )
# )
# print(
#     "sm vs. cm {} logits differ by more then {} decimals out of {}".format(
#         np.count_nonzero(
#             np.round(np.abs(sm_logits - cm_logits), decimals=args.decimals)
#         ),
#         args.decimals,
#         cm_logits.shape[0] * cm_logits.shape[1],
#     )
# )
# print(
#     "sm vs. pm {} logits differ by more then {} decimals out of {}".format(
#         np.count_nonzero(
#             np.round(np.abs(sm_logits - pm_logits), decimals=args.decimals)
#         ),
#         args.decimals,
#         pm_logits.shape[0] * pm_logits.shape[1],
#     )
# )

# # assert np.allclose(sm_logits[0], cm_logits[0], rtol=0.001, atol=0.0)
# # assert np.allclose(sm_logits, cm_logits, rtol=0.1, atol=0.0, )

# # try:
# #     # Make sure cell id's and order match
# #     # Unfortunately the location of the cell ids in h5ad varies...
# #     # assert adata.obs.index.equals(web_predictions.index)
# #     # Make sure predictions and probabilites are close
# #     assert np.all(py_predictions.pred_0.values == web_predictions.pred_0.values)
# #     assert np.all(py_predictions.pred_1.values == web_predictions.pred_1.values)
# #     assert np.all(py_predictions.pred_2.values == web_predictions.pred_2.values)
# #     assert np.allclose(
# #         py_predictions.prob_0.values,
# #         web_predictions.prob_0.values,
# #         atol=args.tolerance,
# #     )
# #     assert np.allclose(
# #         py_predictions.prob_1.values,
# #         web_predictions.prob_1.values,
# #         atol=args.tolerance,
# #     )
# #     assert np.allclose(
# #         py_predictions.prob_2.values,
# #         web_predictions.prob_2.values,
# #         atol=args.tolerance,
# #     )
# # except AssertionError as e:
# #     print(e)
# #     print("Predictions do not match")


# # # Encoder
# # pm = sims.model.network.tabnet.encoder
# # pm.eval()
# # torch.onnx.export(
# #     pm,
# #     torch.zeros(1, len(sims.model.genes)),
# #     raw_model_path,
# #     training=torch.onnx.TrainingMode.EVAL,
# #     input_names=["input"],
# #     output_names=["y0", "y1", "y2", "y3"],
# #     export_params=True,
# #     opset_version=pm_opset_version,
# # )

# # om = onnx.load(raw_model_path)
# # so.list_inputs(om.graph)
# # so.list_outputs(om.graph)

# # pm.eval()  # Set to evaluation mode
# # with torch.no_grad():  # Disable gradient calculation
# #     py = pm.forward(x)

# # oy = so.run(
# #     om.graph,
# #     inputs={"input": x.detach().numpy()},
# #     outputs=["y0", "y1", "y2", "y3"],
# # )

# # oy[0][0][0:4]

# # py[1][0][0].detach().numpy()[0:4]

# # py[0][1][0].detach().numpy()[0:4]
# # py[0][2][0].detach().numpy()[0:4]
