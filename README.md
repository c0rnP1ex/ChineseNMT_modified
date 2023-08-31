# ChineseNMT_modified

Original repo: [hemingkx](https://github.com/hemingkx)/[ChineseNMT](https://github.com/hemingkx/ChineseNMT)
Original dataset: [WMT 2018 Chinese-English track (Only NEWS Area)](https://github.com/hemingkx/ChineseNMT/tree/master/data/json)

Fine-tune dataset: https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/

## Result

**Corpus Bleu comparison**:

| Model              | WMT 2018               | TED Talk               |
| ------------------ | ---------------------- | ---------------------- |
| Transformer        | **27.184087988240112** | 11.328164744670644     |
| Transformer_w_LoRA | 26.089145858659233     | **12.732218072957036** |