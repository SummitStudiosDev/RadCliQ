
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


import torch
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
np.int = int
np.float = float
import pandas as pd
from bert_score import BERTScorer
from radgraph import F1RadGraph
import pickle
from radgraph import F1RadGraph
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn as nn

from .chexbert import bert_encoder, add_semb_col
from .bleu2 import add_bleu_col
from .radgraphcol import add_radgraph_col
from .bertscore import add_bertscore_col


class RadCliQ:
    def __init__(self):
        self.class_file_dir = os.path.dirname(os.path.abspath(__file__))

        self.COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
        self.NORMALIZER_PATH = self.class_file_dir+"/CXR-Report-Metric/CXRMetric/normalizer.pkl"
        self.COMPOSITE_METRIC_V0_PATH = self.class_file_dir+"/CXR-Report-Metric/CXRMetric/composite_metric_model.pkl"
        self.COMPOSITE_METRIC_V1_PATH = self.class_file_dir+"/CXR-Report-Metric/CXRMetric/radcliq-v1.pkl"
        self.BATCH_SIZE = 5

        self.f1radgraph = F1RadGraph(reward_level="simple", model_type= 'radgraph')
        self.bertscorer = BERTScorer(
            model_type="distilroberta-base",
            batch_size=256,
            lang="en",
            rescale_with_baseline=True,
            idf=False)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = bert_encoder(False)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.checkpoint = torch.load(self.class_file_dir+'/CXR-Report-Metric/chexbert.pth')
        self.model.load_state_dict(self.checkpoint['model_state_dict'])


    def get_individual_metrics(self, hyps,refs):
        assert len(hyps) == len(refs)
        df = pd.DataFrame({
            'bleu_score': add_bleu_col(hyps,refs),
            'bertscore': add_bertscore_col(hyps,refs,False, self.bertscorer),
            'semb_score': add_semb_col(hyps, refs, self.tokenizer, self.model, self.BATCH_SIZE, self.device),
            'radgraph_combined': add_radgraph_col(hyps, refs, self.f1radgraph)
        })
        return df
    def calc_radcliq_v0(self, hyps,refs):
        df = self.get_individual_metrics(hyps,refs)
        with open(self.COMPOSITE_METRIC_V0_PATH, "rb") as f:
            composite_metric_v0_model = pickle.load(f)
        with open(self.NORMALIZER_PATH, "rb") as f:
            normalizer = pickle.load(f)
        input_data = np.array(df[self.COLS])
        norm_input_data = normalizer.transform(input_data)
        radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
        return radcliq_v0_scores
    
    def calc_radcliq_v1(self, hyps,refs):
        df = self.get_individual_metrics(hyps,refs)
        with open(self.COMPOSITE_METRIC_V1_PATH, "rb") as f:
            composite_metric_v1_model = pickle.load(f)
        input_data = np.array(df[self.COLS])
        radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
        return radcliq_v1_scores