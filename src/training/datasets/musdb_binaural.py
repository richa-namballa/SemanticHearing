import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    def __init__(self, root, subset="train", sample_rate=44100, segment_seconds=6):

        self.root = os.path.join(root, subset)
        self.tracks = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        self.STEMS = ["vocals", "drums", "bass", "other"]
        self.sample_rate = sample_rate
        self.segment_samples = sample_rate * segment_seconds
        self.num_classes = len(self.STEMS)

    def __len__(self):
        return len(self.tracks) * self.num_classes

    def _load_audio(self, path):

        # load without normalization
        audio, sr = torchaudio.load(path, normalize=False)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return audio

    def _random_crop(self, mixture, target):

        T = mixture.shape[-1]

        if T <= self.segment_samples:
            return mixture, target

        start = random.randint(0, T - self.segment_samples)
        end = start + self.segment_samples

        return mixture[:, start:end], target[:, start:end]

    def __getitem__(self, idx):

        track_idx = idx // self.num_classes
        stem_idx = idx % self.num_classes

        track_name = self.tracks[track_idx]
        stem_name = self.STEMS[stem_idx]

        track_dir = os.path.join(self.root, track_name)

        mixture = self._load_audio(os.path.join(track_dir, "mixture.wav"))
        target = self._load_audio(os.path.join(track_dir, f"{stem_name}.wav"))

        max_val = (torch.max(torch.abs(mixture)) + 1e-6)

        # Peak normalization w.r.t. original mixture peak
        mixture = mixture / max_val
        target = target / max_val

        mixture, target = self._random_crop(mixture, target)

        label_vector = torch.zeros(self.num_classes)
        label_vector[stem_idx] = 1.0

        inputs = {
            "mixture": mixture,
            "label_vector": label_vector,
            "metadata": {
                "track": track_name,
                "stem": stem_name
            }
        }

        return inputs, target

    def to(self, inputs, target, device):
        inputs['mixture'] = inputs['mixture'].to(device)
        inputs['label_vector'] = inputs['label_vector'].to(device)
        target = target.to(device)
        return inputs, target

    def output_to(self, output, device):
        for k, v in output.items():
            output[k] = v.to(device)
        return output

    def output_detach(self, output):
        for k, v in output.items():
            output[k] = v.detach()
        return output

    def collate_fn(self, batch):
        inputs, target = zip(*batch)
        inputs = {
            'mixture': torch.stack([i['mixture'] for i in inputs]),
            'label_vector': torch.stack([i['label_vector'] for i in inputs]),
            'metadata': [i['metadata'] for i in inputs]
        }
        target = torch.stack(target)
        return inputs, target

    def tensorboard_add_metrics(self, writer, tag, metrics, step):
        """
        Add metrics to tensorboard.
        """
        vals = np.asarray(metrics['scale_invariant_signal_noise_ratio'])

        return
