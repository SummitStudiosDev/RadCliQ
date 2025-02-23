#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertModel, AutoModel
import torch.nn as nn
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


# In[1]:


import torch
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
np.int = int
np.float = float
import re
import pandas as pd
 
#Bertscore
from bert_score import BERTScorer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
bertscorer = BERTScorer(
    model_type="distilroberta-base",
    batch_size=256,
    lang="en",
    rescale_with_baseline=True,
    idf=False)


#Radgraph
from radgraph import F1RadGraph
f1radgraph = F1RadGraph(reward_level="simple", model_type= 'radgraph')

#Chexbert
BATCH_SIZE = 5
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = bert_encoder(False)
model = nn.DataParallel(model)
model = model.to(device)
checkpoint = torch.load('CXR-Report-Metric/chexbert.pth')
model.load_state_dict(checkpoint['model_state_dict'])

#BLEU-2
from fast_bleu import BLEU
weights = {"bigram": (1/2., 1/2.)}

#composite 
import pickle
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
NORMALIZER_PATH = "CXR-Report-Metric/CXRMetric/normalizer.pkl"
COMPOSITE_METRIC_V0_PATH = "CXR-Report-Metric/CXRMetric/composite_metric_model.pkl"
COMPOSITE_METRIC_V1_PATH = "CXR-Report-Metric/CXRMetric/radcliq-v1.pkl"



# In[ ]:


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
def gen_embeddings(to_tokenize):
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
def add_semb_col(hyps, refs):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = gen_embeddings(refs)
    pred_embeds = gen_embeddings(hyps)
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


# In[4]:


def add_radgraph_col(hyps, refs):
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)
    return reward_list
    #pred_df["radgraph_combined"] = reward_list
    #return pred_df


# In[5]:


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]
def add_bleu_col(hyps, refs):
    bleu_scores = []
    for i in range(len(hyps)):
        gt_report = prep_reports([refs[i]])[0]
        predicted_report = prep_reports([hyps[i]])[0]
        bleu = BLEU([gt_report], weights)
        score = bleu.get_score([predicted_report])["bigram"]
        bleu_scores.append(score[0])
    return bleu_scores


# In[35]:


def add_bertscore_col(hyps, refs, use_idf):
    """Computes BERTScore and adds scores as a column to prediction df."""
    test_reports = refs
    test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
    method_reports = hyps
    method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

    _, _, f1 = bertscorer.score(method_reports, test_reports)
    #pred_df["bertscore"] = f1
    #return pred_df
    return f1.tolist()


# In[7]:


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


# In[18]:


def get_individual_metrics(hyps,refs):
    assert len(hyps) == len(refs)
    df = pd.DataFrame({
        'bleu_score': add_bleu_col(hyps,refs),
        'bertscore': add_bertscore_col(hyps,refs,False),
        'semb_score': add_semb_col(hyps, refs),
        'radgraph_combined': add_radgraph_col(hyps, refs)
    })
    return df
def calc_radcliq_v0(hyps,refs):
    df = get_individual_metrics(hyps,refs)
    with open(COMPOSITE_METRIC_V0_PATH, "rb") as f:
        composite_metric_v0_model = pickle.load(f)
    with open(NORMALIZER_PATH, "rb") as f:
        normalizer = pickle.load(f)
    input_data = np.array(df[COLS])
    norm_input_data = normalizer.transform(input_data)
    radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
    return radcliq_v0_scores
def calc_radcliq_v1(hyps,refs):
    df = get_individual_metrics(hyps,refs)
    with open(COMPOSITE_METRIC_V1_PATH, "rb") as f:
        composite_metric_v1_model = pickle.load(f)
    input_data = np.array(df[COLS])
    radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
    return radcliq_v1_scores


# In[ ]:

'''
refs = [
    "Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary abnormality",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "no acute cardiopulmonary abnormality",
    "extreme acute cardiopulmonary abnormality",
]
print(get_individual_metrics(hyps,refs))
calculated_radcliq = calc_radcliq_v1(hyps,refs)
#one_over_radcliq = [1/s for s in calculated_radcliq]
print(calculated_radcliq)
'''
