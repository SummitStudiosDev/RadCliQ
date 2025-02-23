def add_radgraph_col(hyps, refs, f1radgraph):
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)
    return reward_list
