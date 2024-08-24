from collections import defaultdict
import inspect
from typing import Dict, List, Optional, Union, Callable, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from torchmetrics.classification.f_beta import F1Score
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerFast, PreTrainedTokenizer, BatchEncoding

from resolve.common.data import BaseSentence, BaseWord
from resolve.common.util import flatten
from resolve.model import COMPONENT_REGISTRY
from resolve.model.components import ItemInstance, MWENegativeSenseCrossProductHead, MeanReducer, CLSReducer, \
    DefinitionOutput, BatchOutput, ModelException, Head
from resolve.training.data import SENSE_MASK_ID, TrainingDefinitionManager


class ContextDictionaryBiEncoder(pl.LightningModule):
    val_f1 = 'val/epoch_f1'
    val_loss = 'val/epoch_loss'
    val_wsd_f1 = 'val/epoch_wsd_f1'
    val_wsd_loss = 'val/epoch_wsd_loss'
    val_mwe_f1 = 'val/epoch_mwe_f1'
    val_mwe_loss = 'val/epoch_mwe_loss'

    loss_metrics = [val_loss, val_wsd_loss, val_mwe_loss]
    f1_metrics = [val_f1, val_wsd_f1, val_mwe_f1]

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=30)
        return [opt], [scheduler]

    def setup_for_train_eval(self, def_manager: TrainingDefinitionManager,
                             mwe_eval: Optional[List['MWEEvalData']] = None):
        # this can't happen in __init__, because when reloading the model we don't know the tokenizers
        # and so can't build the manager until after the model is initialized
        if mwe_eval is None:
            mwe_eval = []
        assert self._def_manager is None, 'def_manager cannot be set twice'
        self._def_manager = def_manager
        self._context_tokenizer = def_manager.context_tokenizer

        self.mwe_eval_pipelines = [
            (eval_data.value, eval_data.get_evaluator(model=self))
            for eval_data in mwe_eval
        ]

        for name, evaluator in self.mwe_eval_pipelines:
            evaluator.pipeline.index.disconnect()

    def set_context_tokenizer(self, tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]):
        self._context_tokenizer = tokenizer

    @staticmethod
    def get_head_from_config(config: Dict[str, Any]) -> Head:
        head_args = {}
        for arg, value in config["head_arguments"].items():
            # may need to initialize a component
            if value in COMPONENT_REGISTRY:
                # todo: reducers with more arguments
                head_args[arg] = COMPONENT_REGISTRY[value]()
            else:
                head_args[arg] = value

        head = COMPONENT_REGISTRY[config["head_name"]](**head_args)
        assert isinstance(head, Head), 'Head generated from config must be an instance of Head class'
        return head

    def __init__(self, encoder: str, lr: float, weight_decay: float, dropout: float,
                 head_config: Dict[str, Any], definition_encoder: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self._def_manager = None
        self._context_tokenizer = None

        model_config = AutoConfig.from_pretrained(encoder, hidden_dropout_prob=dropout)
        self.context_encoder = AutoModel.from_pretrained(encoder, config=model_config)
        self.hidden_size = model_config.hidden_size
        if definition_encoder is None:
            definition_encoder = encoder
        definition_model_config = AutoConfig.from_pretrained(definition_encoder, hidden_dropout_prob=dropout)
        self.definition_encoder = AutoModel.from_pretrained(definition_encoder, config=definition_model_config)
        self.head = self.get_head_from_config(head_config)

        self.valid_f1 = F1Score()
        self.valid_wsd_f1 = F1Score()
        self.valid_mwe_f1 = F1Score()

        self.mwe_eval_pipelines = []

    def on_load_checkpoint(self, checkpoint):
        # See https://github.com/Mindful/MWEasWSD/issues/2
        keys_to_delete = [
            'context_encoder.embeddings.position_ids',
            'definition_encoder.embeddings.position_ids',
        ]
        for key in keys_to_delete:
            if key in checkpoint['state_dict']:
                del checkpoint['state_dict'][key]

    def _compute_item_instances(self, context_ids: Tensor, context_embeddings: Tensor, item_ids: Tensor,
                                sense_keys: Tensor, labels: Optional[Tensor], mwe: bool) -> List[List[ItemInstance]]:
        """Computes item instances for the entire batch, so tensors are of shape B x N or B x K
        where N and K are the number of input token IDs or input items respectively, and B is batch size"""
        item_instances = []
        item_id_list = item_ids.tolist()
        sense_keys = sense_keys.cpu()  # move all to CPU so individual .item() calls don't move data from GPU

        for batch_idx in range(context_ids.shape[0]):
            sentence_item_ids = item_id_list[batch_idx]
            sentence_item_indices = defaultdict(list)

            for idx, item_id in enumerate(sentence_item_ids):
                if item_id != SENSE_MASK_ID:
                    sentence_item_indices[item_id].append(idx)

            item_instances.append([
                ItemInstance(
                    parent=self,
                    context_embedding=context_embeddings[batch_idx],
                    context_ids=context_ids[batch_idx],
                    target_indices=indices,
                    item_id=item_id,
                    sense_key=sense_keys[batch_idx, item_id].item(),
                    mwe=mwe,
                    gold_label=labels[batch_idx, item_id] if labels is not None else None
                ) for item_id, indices in sentence_item_indices.items()
            ])

        return item_instances

    @staticmethod
    def mean_pooling(sbert_output, attention_mask):
        token_embeddings = sbert_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def context_forward(self, sentence_ids: Tensor, sentence_attention_mask: Optional[Tensor] = None) -> Tensor:
        if sentence_attention_mask is None:
            sentence_attention_mask = (sentence_ids != self._context_tokenizer.pad_token_id).int()

        return self.context_encoder(sentence_ids, sentence_attention_mask).last_hidden_state

    def definition_forward(self, definition_batch: BatchEncoding) -> Tensor:
        if self.definition_encoder.name_or_path.startswith('sentence-transformers'):
            batch = definition_batch.copy().to(device=self.device)
            return F.normalize(self.mean_pooling(self.definition_encoder(**batch), batch['attention_mask']), p=2, dim=1)
        # copy the batch because BatchEncoding's to() method moves the base object
        return self.definition_encoder(**definition_batch.copy().to(device=self.device)).last_hidden_state

    def key_to_def_embeddings(self, item_sense_key: int) -> Tensor:
        assert self._def_manager is not None, 'cannot _get_def_embeddings() without setting def_manager'

        # normally pytorch lightning doesn't require to() calls/device management, but the sense inventory does.
        # it's neither part of the input batch nor a registered parameter so lightning doesn't move it (and we don't
        # want it to, that would potentially be wasting a lot of memory) so we just move the required definition batch
        definition_batch = self._def_manager.get_definition_batch(item_sense_key)
        return self.definition_forward(definition_batch)

    def process_single_instance(self, sentence: BaseSentence, words: List[BaseWord],
                                definition_batch: BatchEncoding, mwe: bool) -> DefinitionOutput:
        if any(x is None for x in (getattr(sentence, 'embedding', None), getattr(sentence, 'input_ids', None))):
            raise ModelException('Cannot process sentences without embeddings or input_ids set')

        target_token_indices = [token_idx for token_idx, word_idx in enumerate(sentence.token_word_ids)
                                if word_idx.item() in {w.idx for w in words}]

        definition_embeddings = self.definition_forward(definition_batch)

        item_instance = ItemInstance(
            parent=self,
            context_embedding=sentence.embedding.to(self.device),
            context_ids=sentence.input_ids.to(self.device),
            target_indices=target_token_indices,
            item_id=None,
            sense_key=None,
            mwe=mwe,
            gold_label=None,
            definition_embeddings=definition_embeddings
        )

        return self.head(item_instance)

    def _instances_to_output(self, instances: List[List[ItemInstance]]) -> List[List[DefinitionOutput]]:
        return [
            [self.head(instance) for instance in instance_sublist if instance.is_target]
            for instance_sublist in instances
        ]

    def forward(self, batch: Dict[str, Tensor]) -> BatchOutput:
        encoded_contexts = self.context_forward(batch['sentence_ids'], batch['sentence_attention_mask'])

        if 'word_sense_keys' in batch:
            # word_ids always in batch, so we check for sense keys instead
            word_instances = self._compute_item_instances(batch['sentence_ids'], encoded_contexts, batch['word_ids'],
                                                          batch['word_sense_keys'],
                                                          batch.get('word_sense_labels', None), mwe=False)
            word_outputs = self._instances_to_output(word_instances)
        else:
            word_outputs = None

        if 'mwe_ids' in batch:
            mwe_instances = self._compute_item_instances(batch['sentence_ids'], encoded_contexts, batch['mwe_ids'],
                                                         batch['mwe_sense_keys'],
                                                         batch.get('mwe_sense_labels', None), mwe=True)
            mwe_outputs = self._instances_to_output(mwe_instances)

        else:
            mwe_outputs = None

        return BatchOutput(
            word_outputs,
            mwe_outputs
        )

    def _average_loss(self, outputs: List[DefinitionOutput]) -> torch.Tensor:
        if len(outputs) == 0:
            return torch.tensor(0, device=self.device)
        else:
            return torch.stack([output.loss for output in outputs]).mean()

    def _step(self, batch: Dict[str, Tensor]) -> STEP_OUTPUT:
        model_output: BatchOutput = self(batch)
        step_output = {}

        if model_output.word_outputs is not None:
            flat_word_outputs = flatten(model_output.word_outputs)
            word_loss = self._average_loss(flat_word_outputs)

            step_output['word_loss'] = word_loss.detach()
            step_output['word_pred_labels'] = torch.tensor([output.label_pred for output in flat_word_outputs])
            step_output['word_gold_labels'] = torch.tensor([output.label_gold for output in flat_word_outputs])
        else:
            word_loss = None

        if model_output.mwe_outputs is not None:
            flat_mwe_outputs = flatten(model_output.mwe_outputs)
            mwe_loss = self._average_loss(flat_mwe_outputs)

            step_output['mwe_loss'] = mwe_loss.detach()
            step_output['mwe_pred_labels'] = torch.tensor([output.label_pred for output in flat_mwe_outputs])
            step_output['mwe_gold_labels'] = torch.tensor([output.label_gold for output in flat_mwe_outputs])
        else:
            mwe_loss = None

        if word_loss is not None and mwe_loss is not None and mwe_loss > 0:
            step_output['loss'] = torch.stack((word_loss, mwe_loss)).mean()
        else:
            step_output['loss'] = next(loss for loss in (word_loss, mwe_loss) if loss is not None)

        return step_output

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> STEP_OUTPUT:
        step_output = self._step(batch)

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            step_metrics = {'train/step_loss': step_output['loss']}

            if 'word_loss' in step_output:
                step_metrics['train/step_wsd_loss'] = step_output['word_loss']

            if 'mwe_loss' in step_output:
                # MWE loss is going to be pretty spiky because it will be 0 if we log on a step with no MWEs in it
                step_metrics['train/step_mwe_loss'] = step_output['mwe_loss']

            self.logger.log_metrics(step_metrics, step=self.trainer.global_step)

        return step_output

    def validation_step(self, batch: Dict[str, Tensor], _batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._step(batch)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        for output in outputs:
            if 'word_pred_labels' in output:
                self.valid_f1(output['word_pred_labels'], output['word_gold_labels'])
                self.valid_wsd_f1(output['word_pred_labels'], output['word_gold_labels'])

            if 'mwe_pred_labels' in output and output['mwe_pred_labels'].shape[0] > 0:
                self.valid_f1(output['mwe_pred_labels'], output['mwe_gold_labels'])
                self.valid_mwe_f1(output['mwe_pred_labels'], output['mwe_gold_labels'])

        epoch_loss = sum(output['loss'] for output in outputs)
        self.log(self.val_f1, self.valid_f1)
        self.log(self.val_loss, epoch_loss)

        if 'word_loss' in outputs[0]:
            word_loss = sum(output['word_loss'] for output in outputs)
            self.log(self.val_wsd_f1, self.valid_wsd_f1)
            self.log(self.val_wsd_loss, word_loss)

        if 'mwe_loss' in outputs[0]:
            mwe_loss = sum(output['mwe_loss'] for output in outputs)
            self.log(self.val_mwe_f1, self.valid_mwe_f1)
            self.log(self.val_mwe_loss, mwe_loss)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # this is run after the validation_epoch_end, we run the pipeline eval here so we can still eval the MWE
        # pipeline even if we skip normal eval
        if 'mwe_loss' in outputs[0]:
            self._run_pipeline_eval()

    def _run_pipeline_eval(self):
        for name, evaluator in self.mwe_eval_pipelines:
            evaluator.pipeline.index.connect(in_memory=True)
            aligned_results, binary_results, pipeline_results = evaluator()
            evaluator.pipeline.index.disconnect()
            self.log(f'val/{name}/mwe_pipeline_f1', aligned_results.f1)
            self._log_pipeline_results(pipeline_results['ModelOutputIsMWE'], name)

            if 'ModelFilterOnly' in pipeline_results:
                self._log_pipeline_results(pipeline_results['ModelFilterOnly'], name, 'standalone')

    def _log_pipeline_results(self, results: Dict, name: str, suffix: Optional[str] = None):
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        word_results = results['word']
        self.log(f'val/{name}/mwe_filter{suffix}_acc', word_results['accuracy'])
        self.log(f'val/{name}/mwe_filter{suffix}_tp', word_results['1']['precision'])
        self.log(f'val/{name}/mwe_filter{suffix}_tn', word_results['0']['precision'])
        self.log(f'val/{name}/mwe_filter{suffix}_pos', results['mwe_hyp_pos_rate'])
