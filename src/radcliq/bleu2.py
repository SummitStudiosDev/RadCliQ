weights = {"bigram": (1/2., 1/2.)}
def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]
def add_bleu_col(hyps, refs, BLEU):
    bleu_scores = []
    for i in range(len(hyps)):
        gt_report = prep_reports([refs[i]])[0]
        predicted_report = prep_reports([hyps[i]])[0]
        bleu = BLEU([gt_report], weights)
        score = bleu.get_score([predicted_report])["bigram"]
        bleu_scores.append(score[0])
    return bleu_scores


