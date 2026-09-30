import numpy as np
import pickle
import math
from typing import Any, Mapping, Sequence

class DataLoader:

    def __init__(self, dataset, batch_size=1, randomize=True, replacement=False, shuffle=True, curriculum=None):
        """ Initialize the data loader

        Args:
            dataset: the dataset to load (a dict of inputs and infos)
            batch_size: the number of samples to load in each batch
            randomize: whether to randomize the dataset ordering before loading;
                       if False, the dataset will be loaded in the order it is
                       provided (replacement and shuffle be ignored)
            replacement: whether to sample with replacement
            shuffle: whether to shuffle the dataset after each epoch
        """
        assert isinstance(dataset, dict), "Dataset must be a dict"
        assert 'inputs' in dataset and 'infos' in dataset, "Dataset must have 'inputs' and 'infos' key"
        assert len(dataset['inputs']) == len(dataset['infos']), "Inputs and infos must have the same length"

        if not dataset["inputs"] or type(batch_size) is not int or batch_size < 1:
            raise ValueError("DataLoader requires nonempty data and a positive integer batch_size")
        self.curriculum = CurriculumBuffer(**curriculum) if curriculum is not None else None
        self._last_indices = []
        self.dataset = dataset
        self.batch_size = batch_size
        self.randomize = randomize
        self.replacement = replacement
        self.shuffle = shuffle
        self.n_epochs = -1
        self._i = 0
        self._indices = [ i for i in range(len(self.dataset['inputs'])) ]
        self._exhausted = False
        self._start_new_epoch()  # self.n_epochs will be set to 0

    def _start_new_epoch(self):
        if self.shuffle:
            self._indices = self._update_indices()
        self._i = 0
        self.n_epochs += 1
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next batch of data, always of batch_size. If the dataset is smaller or at the end, the batch will include data from the next epoch after shuffling."""
        self._exhausted = self._exhausted or (self._i >= len(self._indices))
        if self._exhausted:
            self._start_new_epoch()
            raise StopIteration
        xs = []
        infos = []
        self._last_indices = []
        while len(xs) < self.batch_size:
            if self._i >= len(self._indices):
                self._start_new_epoch()
                self._exhausted = True  # Mark as exhausted to stop further sampling in this epoch
            remaining = self.batch_size - len(xs)
            end = min(self._i + remaining, len(self._indices))
            indices = self._indices[self._i:end]
            self._last_indices.extend(int(index) for index in indices)
            xs.extend([self.dataset['inputs'][ind] for ind in indices])
            infos.extend([self.dataset['infos'][ind] for ind in indices])
            self._i += len(indices)
        return xs, infos

    def _update_indices(self):
        N = len(self.dataset['inputs'])
        if self.randomize:
            return np.random.choice(N, size=N, replace=self.replacement)
        else:
            return np.arange(N)

    def sample(self):
        """ Sample a batch of data from the dataset """
        try:
            xs, infos = next(self)
        except StopIteration:
            xs, infos = self.sample()  # make sure to get a batch after resetting
        self._exhausted = False  # calling next() again should not raise StopIteration immediately
        if self.curriculum is not None:
            recent = list(reversed(self.curriculum.history))[:max(0, self.batch_size - 1)]
            fresh = [index for index in self._last_indices if index not in recent]
            fresh += self._last_indices
            self._last_indices = fresh[:self.batch_size - len(recent)] + recent
            xs = [self.dataset["inputs"][index] for index in self._last_indices]
            infos = [self.dataset["infos"][index] for index in self._last_indices]
        return xs, infos

    def observe_scores(self, scores: Sequence[float]) -> None:
        """Record training-only observations for the most recently drawn batch."""
        if self.curriculum is None:
            return
        if len(scores) != len(self._last_indices):
            raise ValueError("curriculum scores must match the last sampled batch")
        self.curriculum.observe(dict(zip(self._last_indices, scores)))

    def __getstate__(self):
        """Get the state of the dataset for pickling."""
        state = self.__dict__.copy()
        state.pop('dataset', None)  # Remove dataset to avoid pickling issues
        return state

    def __setstate__(self, state):
        """Set the state of the dataset from pickling."""
        self.__dict__.update(state)
        # Note: dataset needs to be set manually after unpickling
        print("Warning: dataset needs to be set manually after unpickling.")


class CurriculumBuffer:
    """Bounded recent failed-then-solved example indices, independent of any Guide.

    Success means a finite guide score at least success_threshold. A failure must
    come from an earlier observation batch. Positive observations alone do not add
    examples. Initial data remain available for exploration, avoiding lock-in.
    """

    def __init__(self, *, history_size: int = 2, success_threshold: float = 1.0) -> None:
        """Initialize explicit score semantics and bounded replay memory."""
        if type(history_size) is not int or history_size < 1:
            raise ValueError("curriculum history_size must be a positive integer")
        if not isinstance(success_threshold, (float, int)) or not math.isfinite(success_threshold):
            raise ValueError("curriculum success_threshold must be finite")
        self.history_size = history_size
        self.success_threshold = float(success_threshold)
        self.history: list[int] = []
        self.failed: set[int] = set()
        self.events: list[dict[str, Any]] = []

    def add_success_after_fail(self, index: int) -> None:
        """Remember an observed transition, keeping the newest unique examples."""
        if index not in self.failed:
            return
        self.failed.remove(index)
        if index in self.history:
            self.history.remove(index)
        self.history.append(index)
        self.history = self.history[-self.history_size:]
        self.events.append({"method": "add_success_after_fail", "index": index, "history": list(self.history)})

    def observe(self, scores: Mapping[int, float]) -> None:
        """Register one deterministic observation batch; skip nonfinite failures."""
        for index, score in scores.items():
            if not math.isfinite(score):
                continue
            if score >= self.success_threshold:
                self.add_success_after_fail(index)
            else:
                self.failed.add(index)
