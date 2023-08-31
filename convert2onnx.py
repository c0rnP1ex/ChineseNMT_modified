import torch
from torch.utils.data import DataLoader
from model import Transformer, make_model
from data_loader import MTDataset
import config
import onnx

# load model
model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
model.load_state_dict(torch.load(config.model_path))
model.eval()

input = MTDataset("./data/input.json")
input_dataloader = DataLoader(input, shuffle=False, batch_size=1,
                                 collate_fn=input.collate_fn)

for batch in input_dataloader:
    src = batch.src
    src_mask = batch.src_mask
    src_text = batch.src_text
    trg_text = batch.trg_text
    break

# export model
with torch.no_grad():
    torch.onnx.export(
        model, 
        (batch.src, batch.trg, batch.src_mask, batch.trg_mask), 
        "transformer.onnx", 
        verbose=True, 
        input_names=['src', 'trg', 'src_mask', 'trg_mask'], 
        output_names=["output"]
    )

# check model
onnx_model = onnx.load("transformer.onnx")
onnx.checker.check_model(onnx_model)
print("The model is checked!")




