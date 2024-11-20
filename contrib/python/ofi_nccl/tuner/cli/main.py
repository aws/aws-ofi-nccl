import typer
import pathlib
import pandas as pd
import numpy as np
import os
from enum import Enum
from .wrapper import NCCLProto, NCCLAlgo, Tuner, TunerPlatform

app = typer.Typer(name="show-tuner-decisions")

@app.command()
def show_all(library: pathlib.Path,
             min_ranks_per_node: int = 1,
             max_ranks_per_node: int = 64,
             inc_ranks_per_node: int = 1,
             min_nnodes: int = 2,
             max_nnodes: int = np.log2(2048),
             inc_nnodes: int = 2,
             ):
    df = (
        pd.concat(
            [
                Tuner(library, nranks=(rpn * nodecnt), nnodes=nodecnt, platform=platform).analyze_all()
                for nodecnt in range(min_nnodes, max_nnodes+1, inc_nnodes)
                for rpn in range(min_ranks_per_node, max_ranks_per_node+1, inc_ranks_per_node)
                for platform in TunerPlatform
            ]
        )
        .set_index(["nodes", "ranks", "message_size"])
        .sort_index()
        .dropna()
    )
    df["algo"] = df["algo"].apply(lambda x: str(NCCLAlgo(x).name).title())
    df["proto"] = df["proto"].apply(lambda x: str(NCCLProto(x).name).title())
    df["platform"] = df["platform"].apply(lambda x: str(TunerPlatform(x).name))

    pd.set_option("display.max_rows", None)
    pd.set_option('display.multi_sparse', False)
    for collective in df["collective"].unique():
        sf = df[df["collective"] == collective]
        print(sf)
