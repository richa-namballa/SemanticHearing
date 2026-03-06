import os
import random
import torch
import torchaudio

from semaudio_binaural_base import SemAudioBinauralBaseDataset


class MUSDBBinauralDataset(SemAudioBinauralBaseDataset):

    STEMS = ["vocals", "drums", "bass", "other"]

    def __init__(self, musdb_root, subset="train", segment_seconds=6, sr=44100, **kwargs):
        super().__init__(**kwargs)

        self.root = os.path.join(musdb_root, subset)
        self.tracks = sorted(os.listdir(self.root))

        self.sample_rate = sr
        self.segment_samples = segment_seconds * sr
        self.num_classes = len(self.STEMS)

    def __len__(self):
        return len(self.tracks) * self.num_classes

    def _load_audio(self, path):
        audio, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return audio

    def _crop(self, mixture, target):

        T = mixture.shape[-1]

        if T <= self.segment_samples:
            return mixture, target

        start = random.randint(0, T - self.segment_samples)
        end = start + self.segment_samples

        return mixture[:, start:end], target[:, start:end]

    def __getitem__(self, idx):

        track_idx = idx // self.num_classes
        stem_idx = idx % self.num_classes

        track = self.tracks[track_idx]
        stem = self.STEMS[stem_idx]

        track_dir = os.path.join(self.root, track)

        mixture = self._load_audio(os.path.join(track_dir, "mixture.wav"))
        target = self._load_audio(os.path.join(track_dir, f"{stem}.wav"))

        mixture, target = self._crop(mixture, target)

        label_vector = torch.zeros(self.num_classes)
        label_vector[stem_idx] = 1

        inputs = {
            "mixture": mixture,
            "label_vector": label_vector,
            "metadata": {
                "track": track,
                "stem": stem
            }
        }

        gt = target

        return inputs, gt
