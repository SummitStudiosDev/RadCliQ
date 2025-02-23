import re
def add_bertscore_col(hyps, refs, use_idf, bertscorer):
    """Computes BERTScore and adds scores as a column to prediction df."""
    test_reports = refs
    test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
    method_reports = hyps
    method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

    _, _, f1 = bertscorer.score(method_reports, test_reports)
    #pred_df["bertscore"] = f1
    #return pred_df
    return f1.tolist()
