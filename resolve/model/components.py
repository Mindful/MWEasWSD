from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict

from resolve.common.data import SENSE_MASK_ID


class ModelException(Exception):
    pass



@dataclass
class DefinitionOutput:
    # probabilities associated with each definition
    definition_probs: torch.Tensor
    is_mwe: Optional[bool] = None
    mwe_score: Optional[float] = None

    loss: Optional[torch.Tensor] = None
    # one possible label for each definition, plus one for the "not MWE" case in the case of MWE output
    label_pred: Optional[int] = None
    label_gold: Optional[int] = None
    label_scores: Optional[torch.Tensor] = None

    def cpu(self):
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.cpu())

        return self


@dataclass(frozen=True)
class BatchOutput:
    """Outputs only present if that type of item is being processed (WSD/MWE).
    One definition output per item, so output matrices are B x I, where B is batch size
    and I is the number of items of that type in the given batch sentence"""
    word_outputs: Optional[List[List[DefinitionOutput]]]
    mwe_outputs: Optional[List[List[DefinitionOutput]]]


class ItemInstance:
    def __init__(self, parent: 'ContextDictionaryBiEncoder', mwe: bool,
                 context_embedding: torch.Tensor, context_ids: torch.Tensor,
                 target_indices: List[int], item_id: Optional[int], sense_key: Optional[int],
                 gold_label: Optional[torch.Tensor] = None, definition_embeddings: Optional[torch.Tensor] = None):
        self._parent = parent
        self._definition_embeddings = definition_embeddings
        self.context_embedding = context_embedding
        self.context_ids = context_ids
        self.target_indices = target_indices
        self.item_id = item_id
        self.sense_key = sense_key
        self.mwe = mwe
        self.gold_label = gold_label

    def get_definition_embeddings(self) -> torch.Tensor:
        if self._definition_embeddings is not None:
            return self._definition_embeddings

        assert self.is_target, 'Cannot get definition embeddings for non-target item instance'
        return self._parent.key_to_def_embeddings(self.sense_key)

    @property
    def is_target(self) -> bool:
        return self.sense_key != SENSE_MASK_ID


class Head(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, target: ItemInstance, **kwargs) -> DefinitionOutput:
        raise NotImplementedError()


class ItemReducer(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, target: ItemInstance, **kwargs) -> DefinitionOutput:
        raise NotImplementedError()


class MeanReducer(ItemReducer):
    def forward(self, target: ItemInstance, **kwargs) -> torch.Tensor:
        return torch.mean(target.context_embedding[target.target_indices], dim=0)


class DefinitionReducer(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, target: ItemInstance, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class CLSReducer(DefinitionReducer):
    def forward(self, target: ItemInstance, **kwargs) -> torch.Tensor:
        return target.get_definition_embeddings()[:, 0, :]  # return only the CLS token for each embedding


class IdentityReducer(DefinitionReducer):
    def forward(self, target: ItemInstance, **kwargs) -> torch.Tensor:
        return target.get_definition_embeddings()  # don't reduce definitions


class MWENegativeSenseHead(Head, ABC):
    def __init__(self, hidden_size: int, item_reducer: ItemReducer, definition_reducer: DefinitionReducer):
        super().__init__()
        # this embedding is unused in the case where we aren't training with MWE specific logic
        not_mwe_tensor = torch.empty(1, hidden_size)
        torch.nn.init.xavier_uniform_(not_mwe_tensor)
        self.not_mwe_embedding = torch.nn.parameter.Parameter(not_mwe_tensor)
        self.item_reducer = item_reducer
        self.definition_reducer = definition_reducer

    @classmethod
    def _get_mwe_tuple(cls, label_scores: torch.Tensor) -> Tuple[bool, float]:
        # this is its own method because it has nothing to do with loss computation
        # it's purely to get MWE label data for use in the pipeline
        top_two_scores, top_two_indices = label_scores.topk(2)

        negative_index = label_scores.shape[0] - 1
        top_index = top_two_indices[0].item()
        if top_index == negative_index:
            # the "not MWE" sense is most likely, so we judge it not to be an MWE
            is_mwe = False
            # most likely non-negative def score is score
            mwe_score = torch.softmax(top_two_scores, 0)[1]
        else:
            is_mwe = True
            mwe_score = torch.softmax(label_scores[[negative_index, top_index]], 0)[1]

        return is_mwe, mwe_score

    @classmethod
    def _build_output(cls, target: ItemInstance, label_scores: torch.Tensor) -> DefinitionOutput:
        if target.mwe:
            definition_probs = torch.softmax(label_scores[:-1], 0)
            is_mwe, mwe_score = cls._get_mwe_tuple(label_scores)
        else:
            definition_probs = torch.softmax(label_scores, 0)
            is_mwe = mwe_score = None

        label_pred = label_scores.topk(1).indices
        if target.gold_label is not None:
            loss = torch.nn.functional.cross_entropy(label_scores, target.gold_label)
            return DefinitionOutput(definition_probs, is_mwe, mwe_score,
                                    loss, label_pred, target.gold_label, label_scores)
        else:
            return DefinitionOutput(definition_probs, is_mwe, mwe_score, None, label_pred)


class MWENegativeSenseCrossProductHead(MWENegativeSenseHead):
    def forward(self, target: ItemInstance, **kwargs) -> DefinitionOutput:
        item_embedding = self.item_reducer(target, **kwargs)
        definition_embeddings = self.definition_reducer(target, **kwargs)

        if target.mwe:
            def_embeddings_with_negative = torch.cat((definition_embeddings, self.not_mwe_embedding))
            label_scores = item_embedding @ def_embeddings_with_negative.transpose(0, 1)
        else:
            label_scores = item_embedding @ definition_embeddings.transpose(0, 1)

        return self._build_output(target, label_scores)


@dataclass
class CodeEmbeddingsOutput:
    standard_codes: torch.nn.Module = None
    target_codes: torch.nn.Module = None


class CodeEmbeddings(torch.nn.Module):

    def __init__(self, num_codes: int, hidden_size: int, mwe: bool, poly_type: str):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_size = hidden_size
        self.mwe = mwe
        self.poly_type = poly_type
        self.code_embeddings_dictionary = ModuleDict(self.initialize_code_embeddings())

    def initialize_code_embeddings(self):
        """
        Here we're just initializing all the embeddings we'll need depending on if we want
        to include mwes &/ make a distinction between target * non-target codes
        """
        code_embeddings_dictionary = {}
        if self.poly_type == 'distinct_codes':
            code_embeddings_dictionary['target'] = Embedding(self.num_codes, self.hidden_size)
            torch.nn.init.xavier_normal_(code_embeddings_dictionary['target'].weight)
            if self.mwe:
                code_embeddings_dictionary['mwe_target'] = Embedding(self.num_codes, self.hidden_size)
                torch.nn.init.xavier_normal_(code_embeddings_dictionary['mwe_target'].weight)

        if self.mwe:
            code_embeddings_dictionary['mwe_standard'] = Embedding(self.num_codes, self.hidden_size)
            torch.nn.init.xavier_normal_(code_embeddings_dictionary['mwe_standard'].weight)

        code_embeddings_dictionary['standard'] = Embedding(self.num_codes, self.hidden_size)
        torch.nn.init.xavier_normal_(code_embeddings_dictionary['standard'].weight)

        return code_embeddings_dictionary

    def _get_code_embeddings(self, batch_size:int, embedding_type: str):
        code_ids = torch.arange(self.num_codes, dtype=torch.long, device='cpu' if not torch.cuda.is_available() else "cuda:0")  # (1, num_codes)
        code_ids = code_ids.unsqueeze(0).expand(batch_size, self.num_codes)  # (batch_size, num_codes)
        return self.code_embeddings_dictionary[embedding_type](code_ids)  # (batch_size, num_codes, hidden)

    def forward(self, target: ItemInstance, **kwargs) -> CodeEmbeddingsOutput:
        prefix = 'mwe_' if target.mwe else ''
        
        standard_embeddings = self._get_code_embeddings(kwargs['batch_size'], f'{prefix}standard')
        target_embeddings = self._get_code_embeddings(kwargs['batch_size'], f"{prefix}target") if self.poly_type == 'distinct_codes' else None
        return CodeEmbeddingsOutput(standard_embeddings, target_embeddings)


class DistinctCodesAttention(ItemReducer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def distinct_code_attention(context_embeddings: Tensor,
                                target_indicies: List[int],
                                target_code_embeddings: Tensor,
                                standard_code_embeddings: Tensor,
                                sanity_checks=False) -> Tensor:
        """
        Here we'll do the first attention pass using codes & context.
        At this point we already have code_embeddings corresponding to mwe/single-word, that is
        dealt with in self.code_code_embeddings
        """

        sentence_length = context_embeddings.shape[1]
        # we need to compute this mask here, because word_mask may have other indicies as True
        # that aren't the **current** target
        target_attention_mask = torch.zeros(context_embeddings.shape, device=context_embeddings.device).squeeze()
        target_attention_mask[target_indicies] = 1
        target_context_embeddings = context_embeddings * target_attention_mask

        # invert the 0s and 1s to get the nontarget mask
        nontarget_attention_mask = target_attention_mask.clone()
        nontarget_attention_mask[nontarget_attention_mask == 0] = 1
        nontarget_attention_mask[target_indicies] = 0
        nontarget_context_embeddings = context_embeddings * nontarget_attention_mask

        # intermediate QK_T for targetes & non targets
        intermediate_target_output = (target_code_embeddings @ torch.transpose(target_context_embeddings, 2, 1))

        intermediate_nontarget_output = (standard_code_embeddings @
                                         torch.transpose(nontarget_context_embeddings, 2, 1))

        # now add the two intermeddiates together and finish the attention using the full context as V
        qkt = intermediate_target_output + intermediate_nontarget_output

        code_attended_context = torch.softmax(qkt, -1) @ context_embeddings

        if sanity_checks:
            # cols for target indicies should be nonzero
            assert (all(torch.all(intermediate_target_output[0, :, target_indicies], dim=1)))
            # cols for nontarget indicies should be zero
            assert (not any(torch.any(intermediate_target_output[0, :,
                                      [i for i in range(sentence_length) if i not in target_indicies]], dim=1)))
            # cols for target indicies should be zero here
            assert (all(torch.all(intermediate_nontarget_output[0, :,
                                  [i for i in range(sentence_length) if i not in target_indicies]], dim=1)))
            # cols for non-target indicies should be nonzero here
            assert (not any(torch.any(intermediate_nontarget_output[0, :, target_indicies], dim=1)))

        return code_attended_context

    def forward(self, target: ItemInstance, **kwargs) -> Tensor:
        return self.distinct_code_attention(context_embeddings=target.context_embedding.unsqueeze(0),
                                            target_indicies=target.target_indices,
                                            target_code_embeddings=kwargs["code_embeddings"].standard_codes,
                                            standard_code_embeddings=kwargs["code_embeddings"].target_codes
                                            )


class StandardContextDefinitionAttention(ItemReducer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        # softmax(QK_t) * V
        return torch.softmax(queries @ torch.transpose(keys, 2, 1), -1) @ values

    def forward(self, target: ItemInstance, **kwargs) -> Optional[Tensor]:
        """Here we are using the definition embeddings to compute new output for scores
        """
        definition_embeddings = kwargs.pop("definition_embeddings")
        if definition_embeddings is None:
            return None
        # context features will be (m x hidden_size), definition_cls_batch will be (#-definitions, hidden_sze)
        # so we need to add a dim for # definitions

        num_definitions = definition_embeddings.shape[0]
        num_codes = target.context_embedding.shape[1]
        hidden_size = definition_embeddings.shape[-1]
        # (words, num_codes, hidden)
        code_attended_item = target.context_embedding.expand(num_definitions, num_codes, hidden_size)
        # let definition embedding attend over the context features
        attended_context = self.attention(queries=definition_embeddings,
                                          keys=code_attended_item,
                                          values=code_attended_item)

        # ((attended_context[0] @ definition_embedding.T) * torch.eye(definition_embedding.shape[0])).diagonal(0)
        # is same. Maybe its faster?

        return attended_context


class StandardCodeContextAttention(ItemReducer):
    def __init__(self):
        super().__init__()

    def forward(self, target: ItemInstance, **kwargs) -> Tensor:
        assert(target.context_embedding.shape[0] != 1)  # this should be the entire context, not the target embedding
        return torch.softmax(kwargs["code_embeddings"].standard_codes @ torch.transpose(target.context_embedding.unsqueeze(0), 2, 1),
                             -1) @ target.context_embedding


class TargetCodeQuery(ItemReducer):
    """Target Code Query Reducer.

    This reducer is for an experiment where we multiply code embeddings by the target word's embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, target: ItemInstance, **kwargs) -> CodeEmbeddingsOutput:

        code_embeddings = kwargs["code_embeddings"]
        [batch_size, num_codes, hidden_size] = code_embeddings.standard_codes.shape

        expanded_target_word_embedding = target.context_embedding[target.item_id].expand((num_codes, hidden_size)).expand((batch_size, num_codes, hidden_size))

        return CodeEmbeddingsOutput(standard_codes=code_embeddings.standard_codes * expanded_target_word_embedding,
                                    target_codes=code_embeddings.target_codes if code_embeddings.target_codes is None else code_embeddings.target_codes * expanded_target_word_embedding
                                    )


class ContextWindowReduction(ItemReducer):
    """Context Window Reducer.

    This reducer is for an experiment where we only allow a window of context around the given
    target word/mwe through to the remainder of the head.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def apply_context_window_to_embeddings(sentence_embedding: Tensor,
                                           target_indices: [int],
                                           context_window: int):

        earliest_target_token = min(target_indices)  # earliest token
        latest_target_token = max(target_indices)  # latest token

        # 0 is fine, we drop CLS in reduce_item_embedding
        lower_bound = max(0, earliest_target_token - context_window)
        # max_len is fine, we drop SEP in reduce_item_embedding
        upper_bound = min(len(sentence_embedding) - 1, latest_target_token + context_window)

        context_indices = torch.arange(lower_bound, upper_bound + 1, step=1)

        return sentence_embedding[context_indices]

    def forward(self, target: ItemInstance, **kwargs) -> Tensor:
        target.context_embedding = self.apply_context_window_to_embeddings(sentence_embedding=target.context_embedding,
                                                       target_indices=target.target_indices,
                                                       context_window=2)
        return target


class PolyEncoder(MWENegativeSenseHead):

    def __init__(self,
                 poly_type: str,
                 num_codes: int,
                 hidden_size: int,
                 mwe_processing: bool,
                 context_reduction: Optional[ItemReducer] = None,
                 code_reduction: Optional[ItemReducer] = None,
                 code_context_attention: Optional[ItemReducer] = StandardCodeContextAttention(),
                 context_definition_attention: Optional[ItemReducer] = StandardContextDefinitionAttention(),
                 item_reducer: Optional[ItemReducer] = MeanReducer(),
                 definition_reducer: Optional[DefinitionReducer] = CLSReducer()
                 ):
        """This head is used for our poly encoder experiments.

        Original poly encoder paper: https://arxiv.org/abs/1905.01969

        :param poly_type: What type of poly encoder is this? Needed for CodeEmbeddings.
        :param num_codes: How many codes do you want?
        :param hidden_size: What is the hidden dimension size?
        :param mwe_processing: Are we processing mwes?
        :param context_reduction: Any transformation we do on just the context i.e. context window
        :param code_context_attention: The first attention application in Poly Encoder
        :param context_definition_attention: The final attention application in Poly Encoder
        """
        super().__init__(hidden_size=hidden_size, item_reducer=item_reducer, definition_reducer=definition_reducer)

        self.code_embeddings = CodeEmbeddings(num_codes=num_codes,
                                              poly_type=poly_type,
                                              hidden_size=hidden_size,
                                              mwe=mwe_processing)

        self.context_reduction = context_reduction
        self.code_reduction = code_reduction
        self.code_context_attention = code_context_attention
        self.context_definition_attention = context_definition_attention

    @staticmethod
    def attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        # softmax(QK_t) * V
        return torch.softmax(queries @ torch.transpose(keys, 2, 1), -1) @ values

    def forward(self, target: ItemInstance, **kwargs) -> DefinitionOutput:

        definition_embeddings = self.definition_reducer(target, **kwargs)
        if target.mwe:
            definition_embeddings = torch.cat((definition_embeddings, self.not_mwe_embedding))

        code_embeddings = self.code_embeddings(target, batch_size=definition_embeddings.shape[0])
        if self.code_reduction:
            code_embeddings = self.code_reduction(target, code_embeddings=code_embeddings)

        if self.context_reduction is not None:
            # todo: edit instance here?
            target = self.context_reduction(target)

        target.context_embedding = self.code_context_attention(target, code_embeddings=code_embeddings)

        attended_context = self.context_definition_attention(target, definition_embeddings=definition_embeddings)

        label_scores = (attended_context * definition_embeddings.unsqueeze(0)).sum(-1).diagonal()

        return self._build_output(target, label_scores)
