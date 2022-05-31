# Zero-Shot and Translation Experiments on XQuAD with mBERT

## [XQuAD-XTREME Dataset](https://huggingface.co/datasets/juletxara/xquad_xtreme)

XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering
performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set
of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German,
Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi and Romanian. Consequently, the dataset is entirely parallel across 12 languages.

We also include "translate-train", "translate-dev", and "translate-test"
splits for each non-English language from XTREME (Hu et al., 2020). These can be used to run XQuAD in the "translate-train" or "translate-test" settings.

As the dataset is based on SQuAD v1.1, there are no unanswerable questions in the data. We chose this
setting so that models can focus on cross-lingual transfer.

We show the average number of tokens per paragraph, question, and answer for each language in the
table below. The statistics were obtained using [Jieba](https://github.com/fxsjy/jieba) for Chinese
and the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)
for the other languages. 

|           |   en  |   es  |   de  |   el  |   ru  |   tr  |   ar  |   vi  |   th  |   zh  |   hi  |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Paragraph | 142.4 | 160.7 | 139.5 | 149.6 | 133.9 | 126.5 | 128.2 | 191.2 | 158.7 | 147.6 | 232.4 |
| Question  |  11.5 |  13.4 |  11.0 |  11.7 |  10.0 |  9.8  |  10.7 |  14.8 |  11.5 |  10.5 |  18.7 |
| Answer    |  3.1  |  3.6  |  3.0  |  3.3  |  3.1  |  3.1  |  3.1  |  4.5  |  4.1  |  3.5  |  5.6  |

## Baselines

We show results using baseline methods in the tables below. We directly fine-tune [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)
and [XLM-R Large](https://arxiv.org/abs/1911.02116) on the English SQuAD v1.1 training data
and evaluate them via zero-shot transfer on the XQuAD test datasets. For translate-train, 
we fine-tune mBERT on the SQuAD v1.1 training data, which we automatically translate
to the target language. For translate-test, we fine-tune [BERT-Large](https://arxiv.org/abs/1810.04805)
on the SQuAD v1.1 training set and evaluate it on the XQuAD test set of the target language,
which we automatically translate to English. Note that results with translate-test are not directly
comparable as we drop a small number (less than 3%) of the test examples.

F1 scores:

| Model                 | en   | ar   | de   | el   | es   | hi   | ru   | th   | tr   | vi   | zh   | ro   | avg  |
|-----------------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| mBERT                 | 83.5 | 61.5 | 70.6 | 62.6 | 75.5 | 59.2 | 71.3 | 42.7 | 55.4 | 69.5 | 58.0 | 72.7 | 65.2 |
| XLM-R Large           | 86.5 | 68.6 | 80.4 | 79.8 | 82.0 | 76.7 | 80.1 | 74.2 | 75.9 | 79.1 | 59.3 | 83.6 | 77.2 |
| Translate-train mBERT | 83.5 | 68.0 | 75.6 | 70.0 | 80.2 | 69.6 | 75.0 | 36.9 | 68.9 | 75.6 | 66.2 | -    | 70.0 |
| Translate-test BERT-L | 87.9 | 73.7 | 79.8 | 79.4 | 82.0 | 74.9 | 79.9 | 64.6 | 67.4 | 76.3 | 73.7 | -    | 76.3 |

EM scores:

| Model                 | en   | ar   | de   | el   | es   | hi   | ru   | th   | tr   | vi   | zh   | ro   | avg  |
|-----------------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| mBERT                 | 72.2 | 45.1 | 54.0 | 44.9 | 56.9 | 46.0 | 53.3 | 33.5 | 40.1 | 49.6 | 48.3 | 59.9 | 50.3 |
| XLM-R Large           | 75.7 | 49.0 | 63.4 | 61.7 | 63.9 | 59.7 | 64.3 | 62.8 | 59.3 | 59.0 | 50.0 | 69.7 | 61.5 |
| Translate-train mBERT | 72.2 | 51.1 | 60.7 | 53.0 | 63.1 | 55.4 | 59.7 | 33.5 | 54.8 | 56.2 | 56.6 | -    | 56.0 |
| Translate-test BERT-L | 77.1 | 58.8 | 66.7 | 65.5 | 68.4 | 60.1 | 66.7 | 50.0 | 49.6 | 61.5 | 59.1 | -    | 62.1 |

## Results

|                      | ar          | de          | zh          | vi          | en          | es          | hi          | el          | th          | tr          | ru          | ro          | avg         |
|:---------------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|:------------|
| ZS_mbert             | 57.8 / 42.2 | 72.6 / 55.9 | 58.2 / 47.3 | 68.1 / 47.9 | 85.0 / 73.5 | 76.4 / 58.1 | 55.3 / 40.6 | 62.2 / 45.2 | 35.1 / 26.3 | 51.1 / 34.9 | 71.3 / 54.7 | 72.4 / 59.5 | 63.8 / 48.8 |
| ZS_xml_r             | 67.9 / 52.1 | 75.3 / 59.8 | 65.0 / 55.0 | 73.6 / 54.5 | 84.4 / 73.8 | 77.0 / 59.2 | 69.0 / 52.5 | 74.3 / 57.0 | 68.0 / 56.4 | 68.0 / 51.8 | 75.1 / 58.6 | 80.0 / 66.3 | 73.1 / 58.1 |
| ZS_xml_r_large       | 75.0 / 58.0 | 79.9 / 63.8 | 66.8 / 58.0 | 79.0 / 59.3 | 86.5 / 75.9 | 81.0 / 62.7 | 76.0 / 60.8 | 79.1 / 61.3 | 72.8 / 61.7 | 74.1 / 58.3 | 80.3 / 63.1 | 83.5 / 70.2 | 77.8 / 62.8 |
| TT_mbert             | 70.4 / 55.8 | 76.7 / 63.3 | 70.1 / 56.6 | 70.6 / 55.6 | nan         | 78.7 / 65.1 | 70.6 / 55.8 | 76.0 / 61.9 | 60.0 / 45.9 | 61.6 / 42.7 | 76.6 / 63.1 | nan         | 71.2 / 56.6 |
| TT_bert              | 69.4 / 55.0 | 75.7 / 62.7 | 69.9 / 56.0 | 72.2 / 58.3 | nan         | 77.2 / 62.6 | 69.7 / 53.7 | 75.0 / 60.6 | 60.5 / 46.5 | 59.9 / 41.8 | 74.9 / 60.5 | nan         | 70.4 / 55.8 |
| TT_bert_large        | 73.6 / 59.1 | 80.4 / 66.4 | 74.0 / 59.5 | 76.4 / 62.1 | nan         | 81.9 / 68.7 | 75.3 / 61.7 | 80.2 / 66.8 | 67.5 / 53.9 | 66.3 / 47.3 | 80.1 / 67.0 | nan         | 75.6 / 61.2 |
| TT_xml_r             | 70.4 / 56.5 | 79.0 / 65.8 | 71.1 / 57.4 | 73.0 / 58.4 | nan         | 79.3 / 66.4 | 72.4 / 57.6 | 77.8 / 65.0 | 60.3 / 45.4 | 63.4 / 44.3 | 77.4 / 63.6 | nan         | 72.4 / 58.0 |
| TT_xml_r_large       | 72.9 / 59.1 | 80.1 / 66.6 | 73.6 / 58.8 | 75.1 / 61.5 | nan         | 81.5 / 67.1 | 74.2 / 60.1 | 79.6 / 66.2 | 61.7 / 46.0 | 66.2 / 48.2 | 79.7 / 65.7 | nan         | 74.5 / 59.9 |
| TT_roberta           | 71.6 / 57.0 | 77.0 / 62.4 | 72.4 / 57.9 | 72.4 / 56.6 | nan         | 80.0 / 64.6 | 72.0 / 55.6 | 76.8 / 63.9 | 62.2 / 46.6 | 63.4 / 44.1 | 77.2 / 62.4 | nan         | 72.5 / 57.1 |
| TT_roberta_large     | 74.8 / 61.1 | 80.4 / 67.1 | 74.0 / 59.9 | 76.4 / 62.0 | nan         | 83.1 / 69.4 | 75.1 / 61.0 | 80.8 / 68.0 | 65.3 / 51.0 | 66.0 / 46.9 | 81.2 / 68.0 | nan         | 75.7 / 61.4 |
| TTr_es_xml_r         | 67.0 / 47.9 | 74.2 / 56.4 | 63.4 / 50.3 | 73.2 / 52.0 | 80.4 / 66.1 | 76.3 / 56.6 | 66.9 / 48.2 | 73.5 / 52.4 | 68.7 / 58.5 | 66.2 / 46.5 | 72.4 / 54.2 | 76.0 / 59.2 | 71.5 / 54.0 |
| TTr_de_xml_r         | 65.9 / 48.2 | 74.3 / 58.8 | 64.7 / 55.0 | 72.7 / 53.2 | 79.8 / 67.1 | 75.9 / 57.9 | 66.4 / 50.6 | 72.3 / 54.4 | 65.4 / 56.8 | 65.8 / 50.8 | 73.1 / 56.4 | 75.3 / 61.1 | 71.0 / 55.9 |
| FT_xquad_mbert       | 90.0 / 84.3 | 94.2 / 90.0 | 87.5 / 84.4 | 93.4 / 87.6 | 97.3 / 95.3 | 96.2 / 92.4 | 88.2 / 77.5 | 92.2 / 87.0 | 25.2 / 16.8 | 89.9 / 84.4 | 94.4 / 90.1 | 95.5 / 91.3 | 87.0 / 81.8 |
| FT_xquad_xlm_r       | 92.5 / 88.2 | 95.1 / 91.8 | 94.0 / 92.9 | 95.5 / 91.3 | 98.5 / 97.5 | 97.8 / 93.6 | 92.6 / 88.6 | 96.0 / 91.8 | 94.0 / 92.4 | 92.0 / 87.3 | 95.2 / 90.8 | 97.7 / 94.8 | 95.1 / 91.8 |
| FT_xquad_xml_r_large | 97.0 / 94.2 | 98.1 / 95.6 | 96.3 / 95.7 | 97.6 / 94.0 | 99.7 / 99.2 | 98.5 / 95.8 | 96.5 / 93.6 | 97.8 / 94.4 | 96.1 / 95.1 | 95.9 / 92.3 | 98.1 / 96.0 | 98.9 / 97.1 | 97.5 / 95.2 |
| data_augm_mbert      | 97.1 / 94.4 | 98.9 / 97.9 | 97.5 / 96.8 | 98.9 / 97.5 | 99.7 / 99.2 | 99.6 / 98.9 | 97.7 / 95.1 | 97.0 / 94.6 | 87.3 / 84.9 | 98.8 / 97.4 | 98.5 / 97.3 | 90.6 / 81.6 | 96.8 / 94.6 |