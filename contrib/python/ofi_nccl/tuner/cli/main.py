import typer
import pathlib
import pandas as pd
import os
from enum import Enum
from .wrapper import NCCLProto, NCCLAlgo, Tuner

app = typer.Typer(name="show-tuner-decisions")


class TunerPlatform(str, Enum):
    P5en = "p5en.48xlarge"
    P5 = "p5.48xlarge"


@app.command()
def show(library: pathlib.Path, platform: TunerPlatform):
    ranks_per_nodes = [1, 2, 4, 8]
    nnodes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    os.environ["OFI_NCCL_FORCE_PRODUCT_NAME"] = platform.value
    df = (
        pd.concat(
            [
                Tuner(library, nranks=(rpn * nodecnt), nnodes=nodecnt).analyze_all()
                for nodecnt in nnodes
                for rpn in ranks_per_nodes
            ]
        )
        .set_index(["nodes", "ranks", "message_size"])
        .sort_index()
        .dropna()
    )
    df["algo"] = df["algo"].apply(lambda x: str(NCCLAlgo(x).name).title())
    df["proto"] = df["proto"].apply(lambda x: str(NCCLProto(x).name).title())

    pd.set_option("display.max_rows", None)
    for collective in df["collective"].unique():
        sf = df[df["collective"] == collective]
        print(sf)
