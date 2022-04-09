import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from transformers import (
    BertModel,
    AutoModel,
    PreTrainedModel,
)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query.unsqueeze(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value).squeeze()

def pool(h, mask, pool_type='max'):
    if pool_type == 'max':
        h = h.masked_fill(mask, -1e9)
        return torch.max(h, 1)[0]
    elif pool_type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        self.model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        # self.model = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
       
        self.conv_2 = nn.Conv1d(config.hidden_size, config.hidden_size, 2, padding=0)
        self.conv_3 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, padding=1)

        self.l0 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.l1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # tricks
        self.l2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.l3 = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.l4 = nn.Linear(config.hidden_size * 8, config.hidden_size)
        # self.l4 = nn.Linear(config.hidden_size * 5, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.l0.weight)
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.l3.weight)
        nn.init.xavier_normal_(self.l4.weight)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        subj_mask=None,
        obj_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        sequence_output = outputs[0]


        # 2-gram
        segment_2 = F.relu(self.conv_2(sequence_output.permute(0, 2, 1)).permute(0, 2, 1))
        pad = torch.zeros(sequence_output.size(0), 1, self.config.hidden_size).to(device)
        segment_2 = torch.cat([segment_2, pad], dim=1)
        # 3-gram
        segment_3 = F.relu(self.conv_3(sequence_output.permute(0, 2, 1)).permute(0, 2, 1))

        subj_output = self.dropout(pool(sequence_output, subj_mask.eq(0).unsqueeze(2)))
        obj_output = self.dropout(pool(sequence_output, obj_mask.eq(0).unsqueeze(2)))
        pooled_output = self.dropout(pooled_output)


        # out layer features
        out_list = [pooled_output]
        # out_list = []
        # hidden and entities to get information for relation 
        query_ent = self.l0(torch.cat([subj_output, obj_output, pooled_output], dim=-1))
        global_out = attention(query_ent, sequence_output, sequence_output, attention_mask.eq(0))
        out_list.append(global_out)

        # mention attention
        h1 = attention(subj_output, sequence_output, sequence_output, attention_mask.eq(0))
        h2 = attention(obj_output, sequence_output, sequence_output, attention_mask.eq(0))
        h0 = attention(pooled_output, sequence_output, sequence_output, attention_mask.eq(0))
        out_list.extend([h0, h1, h2])
        

        # mention query
        query_men = self.l1(torch.cat([h1, h2], dim=-1))
        # mention query segment attention 1
        out_seg1 = attention(query_men, sequence_output, sequence_output, attention_mask.eq(0))
        out_list.append(out_seg1)

        # mention query segment attention 2 and 3
        out_seg2 = attention(query_men, segment_2, segment_2, attention_mask.eq(0))
        out_seg3 = attention(query_men, segment_3, segment_3, attention_mask.eq(0))
        out_list.append(out_seg2)
        out_list.append(out_seg3)

        mix_output = torch.cat(out_list, dim=-1)
        mix_output = self.dropout(torch.relu(self.l4(mix_output)))
    
        logits = self.classifier(mix_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
