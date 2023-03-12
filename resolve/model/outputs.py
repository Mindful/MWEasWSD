from torch import Tensor, softmax


class DefinitionOutput:
    def __init__(self, definition_embeddings: Tensor, label_scores: Tensor, mwe: bool = False):
        self.definition_embeddings = definition_embeddings
        self.label_scores = label_scores
        self.label_probs = softmax(self.label_scores, 0)

        top_mwe_index, top_mwe_score = self.label_probs.topk(1)

        if mwe:
            self.is_mwe = (top_mwe_index != self.label_probs.shape[0] - 1).item()
        else:
            self.is_mwe = None

        self.mwe_score = top_mwe_score.item() if self.is_mwe else 0.0

    def __repr__(self):
        return str(self.__dict__)

    def cpu(self):
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, Tensor):
                setattr(self, attr, val.cpu())

        return self


