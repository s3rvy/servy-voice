import os
from typing import Mapping, MutableMapping

import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from datasets import IterableDataset, Dataset
from tqdm import tqdm
import wget
import tarfile
import shutil

MIT_RIRS_PATH: str = "./models/training/resources/data/mit_rirs"
AUDIOSET_PATH: str = "./models/training/resources/data/audioset"
FREE_MUSIC_ARCHIVE_PATH: str = "./models/training/resources/data/free_music_archive"

def download_room_impules_responses() -> None:
    """
    Download room impulse responses collected by MIT
    https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
    """
    if not os.path.exists(MIT_RIRS_PATH):
        os.mkdir(MIT_RIRS_PATH)
    else:
        print("Room impulse data folder already exists. Assuming data was already downloaded")
        return

    rir_dataset: IterableDataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True)

    # Save clips to 16-bit PCM wav files
    for row in tqdm(rir_dataset):
        name: str = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(
            os.path.join(MIT_RIRS_PATH, name),
            16000,
            (row['audio']['array'] * 32767).astype(np.int16))

def download_audioset() -> None:
    """
    Download noise and background audio

    Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
    # For full-scale training, ownload the entire dataset from
    https://huggingface.co/datasets/agkphysics/AudioSet.
    """
    if not os.path.exists(AUDIOSET_PATH):
        os.mkdir(AUDIOSET_PATH)
    else:
        print("Audioset folder already exists. Assuming data was already downloaded")
        return

    for i in range(0, 10):
        file_name: str = f"bal_train0{i}.tar"
        link: str = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{file_name}"
        wget.download(link, out=AUDIOSET_PATH)
        tarfile.open(f"{AUDIOSET_PATH}/{file_name}", 'r').extractall(f"{AUDIOSET_PATH}/uncompressed")
        os.remove(f"{AUDIOSET_PATH}/{file_name}")

        audioset_dataset: Dataset = datasets.Dataset.from_dict(
            {"audio": [str(i) for i in Path(f"{AUDIOSET_PATH}/uncompressed/audio/bal_train").glob("*.flac")]}
        )
        audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        for i in tqdm(range(0, audioset_dataset.num_rows)):
            try:
                row = audioset_dataset[i]
                name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
                scipy.io.wavfile.write(
                    os.path.join(AUDIOSET_PATH, name),
                    16000,
                    (row['audio']['array']*32767).astype(np.int16))
            except Exception as e:
                print(f"Error occured for file {i} of {file_name}: {e}\n\nContinuing with next file...")
        shutil.rmtree(f"{AUDIOSET_PATH}/uncompressed")

def download_free_music_archive() -> None:
    if not os.path.exists(FREE_MUSIC_ARCHIVE_PATH):
        os.mkdir(FREE_MUSIC_ARCHIVE_PATH)
    else:
        print("Free music archive folder already exists. Assuming data was already downloaded")
        return

    fma_dataset = datasets.load_dataset(
        "benjamin-paine/free-music-archive-small",
        split="train",
        streaming=True)
    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    for row in tqdm(fma_dataset):
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(FREE_MUSIC_ARCHIVE_PATH, name),
            16000,
            (row['audio']['array'] * 32767).astype(np.int16))

if __name__ == "__main__":
    download_room_impules_responses()
    download_audioset()
    download_free_music_archive()