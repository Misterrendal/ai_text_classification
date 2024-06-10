import json
from argparse import ArgumentParser
from pathlib import Path

import more_itertools
import numpy as np
import onnxruntime as ort
import tqdm
import transformers

print(ort.get_device())


def parse():
    parser = ArgumentParser()
    parser.add_argument('onnx_model', type=Path, default=Path('model.onnx'))
    parser.add_argument('--tokenizer', type=Path, default='microsoft/deberta-v3-base')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


class Predictor:
    def __init__(self, onnx_model: Path, tokenizer, batch_size: int):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self._model = ort.InferenceSession(str(onnx_model), providers=['CUDAExecutionProvider'])
        self._batch_size = batch_size

    def get_output_names(self):
        return [model_output.name for model_output in self._model.get_outputs()]

    def get_input_names(self):
        return [model_input.name for model_input in self._model.get_inputs()]

    def tokenize(self, text: str):
        tokens = self._tokenizer(text, padding='max_length', max_length=512, truncation=True)
        return dict(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )

    def forward(self, texts):
        tokens = [self.tokenize(text) for text in texts]
        input_ids = np.array([x['input_ids'] for x in tokens], dtype=np.int32)
        attention_mask = np.array([x['attention_mask'] for x in tokens], dtype=np.int32)
        ort_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        ort_outs = self._model.run(self.get_output_names(), ort_inputs)
        return ort_outs[0], ort_outs[1]

    def __call__(self, texts):
        confs = []
        model_ids = []
        chunks = list(more_itertools.chunked(texts, self._batch_size))
        for batch in tqdm.tqdm(chunks):
            conf, model_idx = self.forward(batch)
            confs.extend(conf)
            model_ids.extend(np.argmax(model_idx, axis=1))
        ai_labels = [float(conf) for conf in confs]
        return ai_labels, model_ids


def load_test_data(sample_filepath: str):
    with open(sample_filepath, 'r') as f:
        data = json.load(f)
        texts = data['texts']
        labels = data['labels']
    return texts, labels


def run(predictor, sample_filepath):
    texts, labels = load_test_data(sample_filepath)
    pred_labels = predictor(texts)
    count = np.sum(np.array(labels) == pred_labels)
    total = len(labels)
    return count, total


def main(args):
    predictor = Predictor(args.onnx_model, args.tokenizer, args.batch_size)

    texts, gt_labels, gt_model_ids = load_data()
    pred_ai_labels, pred_model_ids = predictor(texts)

    accuracy = 100 * np.sum(np.array(gt_labels) == pred_ai_labels) / len(gt_labels)
    print(f'Accuracy ai: {accuracy:.2f}%')

    model_accuracy = 100 * np.sum(np.array(gt_model_ids) == pred_model_ids) / len(gt_model_ids)
    print(f'Accuracy model name: {model_accuracy:.2f}%')


if __name__ == '__main__':
    args = parse()
    main(args)
