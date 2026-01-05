import typer
import pathlib
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from enum import Enum
from .wrapper import NCCLProto, NCCLAlgo, Tuner, TunerPlatform

app = typer.Typer(name="show-tuner-decisions")


def create_tuner_analysis_for_platform(platform_tasks):
    """
    Process all tasks for a single platform in one subprocess.
    platform_tasks: (platform, [(library, nranks, nnodes), ...])
    """
    platform, tasks = platform_tasks

    results = []
    for library, nranks, nnodes in tasks:
        tuner = Tuner(library, nranks=nranks, nnodes=nnodes, platform=platform)
        results.append(tuner.analyze_all())

    return pd.concat(results) if results else pd.DataFrame()

@app.command()
def show_all(library: pathlib.Path,
             min_ranks_per_node: int = 1,
             max_ranks_per_node: int = 64,
             inc_ranks_per_node: int = 1,
             min_nnodes: int = 2,
             max_nnodes: int = np.log2(2048),
             inc_nnodes: int = 2,
             show_channels: bool = False,
             ):
    # Group tasks by platform
    platform_tasks = {}
    for platform in TunerPlatform:
        platform_tasks[platform] = []
    for nodecnt in range(min_nnodes, max_nnodes+1, inc_nnodes):
        for rpn in range(min_ranks_per_node, max_ranks_per_node+1, inc_ranks_per_node):
            for platform in TunerPlatform:
                platform_tasks[platform].append((library, rpn * nodecnt, nodecnt))

    # Convert to list of (platform, tasks) tuples for multiprocessing
    grouped_args = list(platform_tasks.items())

    # Use multiprocessing with one process per platform to avoid
    # re-initialization issues due to a singleton in the underlying plugin
    # library. Each platform gets its own dedicated subprocess.
    with mp.Pool(processes=len(TunerPlatform), maxtasksperchild=1) as pool:
        results = pool.map(create_tuner_analysis_for_platform, grouped_args)

    df = (
        pd.concat(results)
        .set_index(["nodes", "ranks", "message_size"] if not show_channels else ["nodes", "ranks", "message_size", "nchannels"])
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
