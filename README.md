#### Solution for TextGenAdvTrack-2025Spring/AI-text-Detection

TEAM: SAFETEAM

SUPPORT MODELS: argugpt/argugpt-sent/kerasnlp/mage/qwen3-0.6b

> For argugpt/argugpt-sent, you need to download the model weights from Hugging Face and rename the directory to argugpt-roberta/argugpt-roberta-sent. Otherwise, the version check with the latest transformers will fail.
> 
> For kerasnlp, you need to download the weights from [here](https://storage.googleapis.com/kaggle-data-sets/3947266/6987454/compressed/fold0.keras.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250510%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250510T154500Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0ec24f1d137c10c1e6413a79b4b20cd56a459c816f5bd93a8e28c8930fc8e2fb08bbdaf1614dc5b41838021d46609cfb4be81a739aeffa2a46c8d3cf48366753e664e720fe1ad469d46583ef43b17526becac1c5e950b508afefd063fdb42d0a1103d694431204d4e11008bf16ebaab63a40572c30f0110d4efbec6af9f90f79213503d41f536e42d98bcb184923fbe62ba247a7b36ba4e62607c3d16b52a80a0fb7d2eebba86664f93c1a9520201e066b66de7248f1464f7ab85e2798c7b5127db3942ab5592fd2353e9c8811f57975cd61b66bf83b4289bc015d83b1bb3e38bd54b3e8705415cbc330a21919f59f873914bd6342356f89095bc85efe5a912d)


#### Infer
```bash
bash prediction.sh
```
> modify prediction.sh to use different models

#### Evaluate
```bash
bash evaluate.sh
```


#### Result on val set
| Team/Method           | AUC         | Acc       | F1          | Avg Time (s) | Weighted Score |
|-----------------------|-------------|-----------|-------------|--------------|----------------|
| SAFETEAM-argugpt-sent | 0.921256361 | 0.832416667 | 0.848876531 | 0.00021169   | **0.803327693**    |
| SAFETEAM-kerasnlp	    | 0.925162028	| 0.748166667	 | 0.671307374 |	0.000589239	| 0.780218524  |
| SAFETEAM-mage         | 0.895181306 | 0.775833333 | 0.762577229 | 0.000475547  | 0.770621361    |
| SAFETEAM-argugpt      | 0.894721069 | 0.75375   | 0.735143856 | 0.000210714  | 0.763692786    |
| SAFETEAM-qwen3-0.6b   |	0.458446958	| 0.497416667	| 0.66195841 | 0.083883074 |	0.424955133  |


