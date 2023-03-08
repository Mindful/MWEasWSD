from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Embedding


class PolyEncoderModule(torch.nn.Module):

    def __init__(self, hidden_size=768, num_codes=16, mwe_processing=False, target_item_only=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_codes = num_codes
        self.mwe = mwe_processing
        self.target_item_only = target_item_only

        self.code_embeddings = Embedding(self.num_codes, self.hidden_size)
        torch.nn.init.xavier_normal_(self.code_embeddings.weight)

        if self.mwe:
            self.mwe_code_embeddings = Embedding(self.num_codes, self.hidden_size)
            torch.nn.init.xavier_normal_(self.mwe_code_embeddings.weight)

    @staticmethod
    def attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        # softmax(QK_t) * V
        return torch.softmax(queries @ torch.transpose(keys, 2, 1), -1) @ values

    def _apply_codes_to_item(self, item_embedding: Tensor, mwe=False) -> Tensor:
        if self.target_item_only:
            item_embedding = item_embedding.expand(1, 1, self.hidden_size)
        else:
            raise NotImplementedError
        code_ids = torch.arange(self.num_codes, dtype=torch.long, device=item_embedding.device)        # (1, num_codes)
        code_ids = code_ids.unsqueeze(0).expand(item_embedding.size()[0], self.num_codes)  # (batch_size, num_codes)
        code_embeddings = self.code_embeddings(code_ids) if not mwe else self.mwe_code_embeddings(code_ids)  # (batch_size, num_codes, 768)
        item_embedding = self.attention(queries=code_embeddings,  # batch_size x m x 768
                                        keys=item_embedding,
                                        values=item_embedding)

        return item_embedding

    def _poly_forward(self, code_attended_item: Tensor, definition_embedding: Tensor) -> Tuple[Tensor, Tensor]:

        # context features will be (m x hidden_size), definition_cls_batch will be (#-definitions, hidden_sze)
        # so we need to add a dim for # definitions
        num_definitions = definition_embedding.shape[0]
        code_attended_item = code_attended_item.expand(num_definitions, self.num_codes, self.hidden_size)  # (words, num_codes, 768)
        # let definition embedding attend over the context features
        attended_context = self.attention(queries=definition_embedding,
                                          keys=code_attended_item,
                                          values=code_attended_item)[0]
        score = (attended_context * definition_embedding).sum(-1)
        return attended_context, score

    def forward(self, item_embedding: Tensor, definition_embeddings: Tensor, mwe: bool) -> Tuple[Tensor, Tensor]:
        code_item_output = self._apply_codes_to_item(item_embedding=item_embedding, mwe=mwe)

        return self._poly_forward(code_item_output, definition_embeddings)



