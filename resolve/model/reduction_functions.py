from typing import List
import torch
from transformers import PreTrainedTokenizer



def length_wavg(token_embeddings: List[torch.Tensor],
                word_indices: List[List[int]],
                input_ids: List[torch.Tensor],
                tokenizer: PreTrainedTokenizer,
                scaling: str = 'linear',
                **kwargs):

    # Group subword embeddings by word
    grouped_embeddings = [torch.stack([token_embeddings[index] for index in indices]) for indices in word_indices]

    decoded_words = tokenizer.convert_ids_to_tokens(input_ids[0])
    # A list of subword lengths per text-word, i.e. ["foo", "##bar] -> [3, 3]
    subword_lengths_per_word = [[len(decoded_words[index].replace('##', '')) for index in indices] for indices in word_indices]
    if scaling == 'linear':
        weighted_embeddings = [torch.mean(embeddings * torch.transpose(
                                        torch.tensor([lengths], device=embeddings.get_device() if torch.cuda.is_available() else 'cpu'),
                                        0, 1), dim=0) for embeddings, lengths in
                               zip(grouped_embeddings, subword_lengths_per_word)]
        return torch.stack(weighted_embeddings)

    raise NotImplementedError("Scaling method chosen isn't implemented.")


def mean(token_embeddings: List[torch.Tensor],
         word_indices: List[List[int]],
         **kwargs):
    return torch.stack([torch.mean(token_embeddings[indices], dim=0) for indices in word_indices])


def get_reduction_function(function_name: str):
    return eval(f"{function_name}")

