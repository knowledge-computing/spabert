
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import  SequenceClassifierOutput


class PivotEntityPooler(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, hidden_states, pivot_len_list):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the tokens of pivot entity
        
        bsize = hidden_states.shape[0]

        tensor_list = []
        for i in torch.arange(0, bsize):
            pivot_token_full = hidden_states[i, 1:pivot_len_list[i]+1]
            pivot_token_tensor = torch.mean(torch.unsqueeze(pivot_token_full, 0), dim = 1)
            tensor_list.append(pivot_token_tensor)


        batch_pivot_tensor = torch.cat(tensor_list, dim = 0)

        return batch_pivot_tensor
       

class BaselineTypingHead(nn.Module):
    def __init__(self, hidden_size, num_semantic_types):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
        self.seq_relationship = nn.Linear(hidden_size, num_semantic_types)

    def forward(self, pivot_pooled_output):

        pivot_pooled_output = self.dense(pivot_pooled_output)
        pivot_pooled_output = self.activation(pivot_pooled_output)

        seq_relationship_score = self.seq_relationship(pivot_pooled_output)
        return seq_relationship_score


class BaselineForSemanticTyping(nn.Module):

    def __init__(self, backbone, hidden_size, num_semantic_types):
        super().__init__() 

        self.backbone = backbone
        self.pivot_pooler = PivotEntityPooler() 
        self.num_semantic_types = num_semantic_types

        self.cls = BaselineTypingHead(hidden_size, num_semantic_types)


    def forward(
        self,
        input_ids=None,
        position_ids = None,
        pivot_len_list = None,
        attention_mask=None, 
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict = True
    ):

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids = position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        sequence_output = outputs[0]
        pooled_output = self.pivot_pooler(sequence_output, pivot_len_list) 


        type_prediction_score = self.cls(pooled_output)

        typing_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            typing_loss = loss_fct(type_prediction_score.view(-1, self.num_semantic_types), labels.view(-1))

        

        if not return_dict:
            output = (type_prediction_score,) + outputs[2:]
            return ((typing_loss,) + output) if typing_loss is not None else output

        return SequenceClassifierOutput(
            loss=typing_loss,
            logits=type_prediction_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
