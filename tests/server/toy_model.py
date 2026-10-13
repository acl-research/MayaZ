"""A tiny deterministic ONNX graph with the served model's interface (torch-free), for server tests.

Inputs are the arrays mayaz.model.render.collate builds (runtime.INPUT_NAMES); the output is `logits` [B, K]:
each option's score is a seeded weight of its first content token plus half the weight of its last content
token, and padded option slots get -1e4, as in the real model.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mayaz.runtime import INPUT_NAMES

VOCAB = 50368


def build_toy_model(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    inits = [numpy_helper.from_array(rng.normal(size=VOCAB).astype(np.float32), "w_first"),
             numpy_helper.from_array(rng.normal(size=VOCAB).astype(np.float32), "w_last"),
             numpy_helper.from_array(np.array(1, np.int64), "one"),
             numpy_helper.from_array(np.array(0.5, np.float32), "half"),
             numpy_helper.from_array(np.array(-1e4, np.float32), "neg")]
    nodes = [helper.make_node("Add", ["opt_start", "one"], ["first_idx"]),
             helper.make_node("Sub", ["opt_end", "one"], ["last_idx"]),
             helper.make_node("GatherElements", ["input_ids", "first_idx"], ["first_tok"], axis=1),
             helper.make_node("GatherElements", ["input_ids", "last_idx"], ["last_tok"], axis=1),
             helper.make_node("Gather", ["w_first", "first_tok"], ["a"]),
             helper.make_node("Gather", ["w_last", "last_tok"], ["b"]),
             helper.make_node("Mul", ["b", "half"], ["b_half"]),
             helper.make_node("Add", ["a", "b_half"], ["score"]),
             helper.make_node("Where", ["opt_mask", "score", "neg"], ["logits"])]
    bool_in = {"q_mask", "opt_mask"}
    seq_in = {"input_ids", "position_ids", "segment_ids", "q_mask"}
    inputs = [helper.make_tensor_value_info(n, TensorProto.BOOL if n in bool_in else TensorProto.INT64,
                                            ["b", "seq" if n in seq_in else "k"]) for n in INPUT_NAMES]
    out = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", "k"])]
    model = helper.make_model(helper.make_graph(nodes, "mayaz_toy", inputs, out, inits),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path
