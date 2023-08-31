import torch
import torch.nn as nn
import config
from model import Transformer, make_model
import math
DEVICE = config.device

class _LoRALayer(nn.Module):

    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x
    

class LoRA_Transformer(nn.Module):

    def __init__(self, model: Transformer, r: int, d_model: int):
        super(LoRA_Transformer, self).__init__()
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.r = r

        model.load_state_dict(torch.load(config.model_path))

        for param in model.parameters():
            param.requires_grad = False

        for layer in model.encoder.layers:
            w_q_linear = layer.self_attn.linears[0]
            w_v_linear = layer.self_attn.linears[2]
            w_a_linear_q = nn.Linear(d_model, r, bias=False)
            w_b_linear_q = nn.Linear(r, d_model, bias=False)
            w_a_linear_v = nn.Linear(d_model, r, bias=False)
            w_b_linear_v = nn.Linear(r, d_model, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            layer.self_attn.linears[0] = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q).to(DEVICE)
            layer.self_attn.linears[2] = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v).to(DEVICE)
            
        for layer in model.decoder.layers:
            w_q_linear = layer.self_attn.linears[0]
            w_v_linear = layer.self_attn.linears[2]
            w_a_linear_q = nn.Linear(d_model, r, bias=False)
            w_b_linear_q = nn.Linear(r, d_model, bias=False)
            w_a_linear_v = nn.Linear(d_model, r, bias=False)
            w_b_linear_v = nn.Linear(r, d_model, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            layer.self_attn.linears[0] = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q).to(DEVICE)
            layer.self_attn.linears[2] = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v).to(DEVICE)

        self.reset_parameters()
        self.lora_transformer = model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.lora_transformer(src, tgt, src_mask, tgt_mask)
    

if __name__ == '__main__':
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    lora_model = LoRA_Transformer(model, config.r, config.d_model)

    for name, param in lora_model.named_parameters():
        print(f"{name}: {param.shape}")