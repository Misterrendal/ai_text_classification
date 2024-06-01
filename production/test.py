import json
from pathlib import Path

import numpy as np
import requests as re
import tqdm


def load_test_data(sample_filepath: str):
    with open(sample_filepath, 'r') as f:
        data = json.load(f)
        texts = data['texts']
        labels = data['labels']
    return texts, labels


def post(texts, ip):
    response = re.post(f'http://{ip}/predict', json={'list_text': texts})
    return response.json()


if __name__ == '__main__':
    backet_count_1 = 0
    backet_count_2 = 0
    backet_count_mean = 0
    total = 0

    files = list(Path('resources/sample_data/').glob('*.json'))
    for sample_path in tqdm.tqdm(files):
        texts, labels = load_test_data(str(sample_path))

        # answer = post(texts, ip='127.0.0.1:8080')
        answer_1 = post(texts, ip='75.63.212.206:45974')
        answer_2 = post(texts, ip='90.55.27.227:40175')

        pred_labels_1 = np.array(answer_1['result']) > 0.5
        pred_labels_2 = np.array(answer_2['result']) > 0.5
        pred_labels_mean = (np.array(answer_1['result']) + np.array(answer_2['result']) / 2) > 0.5

        backet_count_1 += np.sum(np.array(labels) == pred_labels_1)
        backet_count_2 += np.sum(np.array(labels) == pred_labels_2)
        backet_count_mean += np.sum(np.array(labels) == pred_labels_mean)
        total += len(labels)

        print("Accuracy:",
              f"{(backet_count_1 / total):.4f}",
              f"{(backet_count_2 / total):.4f}",
              f"{(backet_count_mean / total):.4f}")
