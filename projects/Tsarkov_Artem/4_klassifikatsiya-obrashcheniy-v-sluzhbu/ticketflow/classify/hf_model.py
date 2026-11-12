"""Multi-task модель на базе Hugging Face transformers: BERT-энкодер + головы категории и приоритета.

По умолчанию энкодер инициализируется по конфигу (компактный BERT, обучается на корпусе
тикетов вместе с собственным WordPiece-токенизатором). При наличии доступа к HF Hub
можно стартовать с предобученного cointegrated/rubert-tiny2 (параметр init_from).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel


class TicketModel(nn.Module):
    def __init__(self, config: BertConfig, n_cat: int, n_pri: int, init_from: str | None = None) -> None:
        super().__init__()
        self.encoder = BertModel.from_pretrained(init_from) if init_from else BertModel(config)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.cat_head = nn.Linear(h, n_cat)
        self.pri_head = nn.Linear(h, n_pri)
        self.n_cat, self.n_pri = n_cat, n_pri

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = self.drop((out * m).sum(1) / m.sum(1).clamp(min=1))  # mean pooling
        return self.cat_head(pooled), self.pri_head(pooled)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(p)
        torch.save(self.state_dict(), p / "model.pt")
        (p / "heads.json").write_text(json.dumps({"n_cat": self.n_cat, "n_pri": self.n_pri}))

    @classmethod
    def load(cls, path: str | Path) -> "TicketModel":
        p = Path(path)
        heads = json.loads((p / "heads.json").read_text())
        m = cls(BertConfig.from_pretrained(p), heads["n_cat"], heads["n_pri"])
        m.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        return m
