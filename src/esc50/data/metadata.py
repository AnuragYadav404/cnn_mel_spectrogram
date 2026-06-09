from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import soundfile as sf


def enrich_df_with_filepaths(input_df: pd.DataFrame, data_root: str) -> pd.DataFrame:
    output_df = input_df.copy()
    output_df["filepath"] = output_df["filename"].apply(lambda value: os.path.join(data_root, "audio/audio", value))
    output_df["filepath_hq"] = output_df["filename"].apply(lambda value: os.path.join(data_root, "audio/audio/44100", value))
    assert output_df["filepath"].apply(os.path.exists).all()
    assert output_df["filepath_hq"].apply(os.path.exists).all()
    return output_df


def get_audio_metadata(file_path: str) -> Dict[str, object]:
    metadata = sf.info(file_path)
    sample_rate = metadata.samplerate
    num_channels = metadata.channels
    num_frames = metadata.frames
    duration = num_frames / sample_rate if sample_rate else None

    if "PCM_" in metadata.subtype:
        bit_depth = int(metadata.subtype.split("_")[1])
    elif "FLOAT" in metadata.subtype:
        bit_depth = 32
    else:
        bit_depth = None

    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "num_channels": num_channels,
        "bit_depth": bit_depth,
        "encoding": metadata.subtype,
    }


def enrich_df_with_audiometa(input_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [input_df, pd.DataFrame(input_df["filepath"].apply(get_audio_metadata).to_list())],
        axis=1,
    )
