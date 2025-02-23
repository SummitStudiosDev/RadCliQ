from transformers import BertModel, AutoModel
import torch.nn as nn
import torch
import numpy as np
np.int = int
np.float = float

class bert_encoder(nn.Module):
    def __init__(self, logits, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_encoder, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        self.logits = logits
        self.dropout = nn.Dropout(p)
        #size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape
                                            (batch_size, 4) and the last has shape (batch_size, 2)
        """
        #shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        #shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        #cls_hidden = self.dropout(cls_hidden)
        out = cls_hidden
        if self.logits:
            out = []
            for i in range(14):
                out.append(self.linear_heads[i](cls_hidden))
        return out

def tokenize(impressions, tokenizer):
        new_impressions = []
        #print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
        #for i in tqdm(range(len(impressions))):
        for i in range(len(impressions)):
                tokenized_imp = tokenizer.tokenize(impressions[i])
                if tokenized_imp: #not an empty report
                        res = tokenizer.encode_plus(tokenized_imp)['input_ids']
                        if len(res) > 512: #length exceeds maximum size
                                #print("report length bigger than 512")
                                res = res[:511] + [tokenizer.sep_token_id]
                        new_impressions.append(res)
                else: #an empty report
                        new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id]) 
        return new_impressions
def generate_attention_masks(batch, source_lengths, device):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)
def get_imp(idx, encoded_imp):
    imp = encoded_imp[idx]
    imp = torch.LongTensor(imp)
    result = {"imp": imp, "len": imp.shape[0], "idx": idx}
    return result
def collate_fn_no_labels(indicies, encoded_imp):
    sample_list = [get_imp(s, encoded_imp) for s in indicies]
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=0)
    len_list = [s['len'] for s in sample_list]
    idx_list = [s['idx'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list, 'idx': idx_list}
    return batch     
def create_batches(array_length, batch_size):
    batches = []
    for i in range(0, array_length, batch_size):
        batch_indices = list(range(i, min(i + batch_size, array_length)))
        batches.append(batch_indices)
    return batches
def gen_embeddings(to_tokenize, tokenizer, model, BATCH_SIZE, device):
    encoded_imp = tokenize(to_tokenize, tokenizer)
    length_of_data = len(encoded_imp)
    batches = create_batches(length_of_data, BATCH_SIZE)
    rep = {}
    with torch.no_grad():
            #for i in tqdm(batches):
            for i in batches:
                data = collate_fn_no_labels(i, encoded_imp)
                batch = data['imp'] 
                batch = batch.to(device)
                src_len = data['len']
                attn_mask = generate_attention_masks(batch, src_len, device)
                out = model(batch, attn_mask)
                for idx, j in zip(data['idx'], range(len(out))):
                    rep[idx] = out[j].to('cpu')
    return(rep)
#def add_semb_col(pred_df):
def add_semb_col(hyps, refs, tokenizer, model, BATCH_SIZE, device):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = gen_embeddings(refs, tokenizer, model, BATCH_SIZE, device)
    pred_embeds = gen_embeddings(hyps, tokenizer, model, BATCH_SIZE, device)
    list_label_embeds = []
    list_pred_embeds = []
    for data_idx in sorted(label_embeds.keys()):
        list_label_embeds.append(label_embeds[data_idx])
        list_pred_embeds.append(pred_embeds[data_idx])
    np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
    np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    #pred_df["semb_score"] = scores
    #return pred_df
    return scores
