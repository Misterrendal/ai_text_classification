import json
from typing import Tuple

from torch.utils.data import Dataset


class JsonDataset(Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, 'r') as f:
            self._df = json.load(f)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx) -> Tuple[str, Tuple[str, float]]:
        text = self._df['texts'][idx]
        is_ai_generated = self._df['labels'][idx]
        return text, (0, float(is_ai_generated))


if __name__ == '__main__':
    dataset = JsonDataset('resources/sample_data_1715832347273793558.json', model_names=[])
    print(dataset[0])