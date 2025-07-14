def get_match_cat_acc(match_df, multid_thr):
    """
    Given the passed match dataframe, split by the multi_d_err property around
    the multid_thr threshold value.

    Return the accepted matches.

    Args:
        match_df (pd.DataFrame): Candidate match DataFrame with calculated multi_d_err
            column which will be used to accept/reject
    """

    match_acc_df = match_df.loc[match_df["multi_d_err"] < multid_thr]

    return match_acc_df


def count_match_cat_rej(match_df, multid_thr):
    """
    Count the number of matches in match_df with a multi_d_err above threshold.

    Args:
        match_df (pd.DataFrame): Candidate match DataFrame with calculated multi_d_err
            column which will be used to accept/reject
    """
    match_rej_df = match_df.loc[match_df["multi_d_err"] >= multid_thr]

    return len(match_rej_df.index)


def get_acc_series(sub_s, truth_s, scaling):
    """
    Calculate a generic accuracy series based on passed measured and truth values
    according to:
        acc_s = abs(sub_s - truth_s) / scaling

    Scaling can either be a scalar or an iterable of the same length.

    Args:
        sub_s (:obj:`pandas.Series`): Submitted (measured) value series.
        truth_s (:obj:`pandas.Series`): True value series.
        scaling (:obj:`pandas.Series` or :obj:`float64`): Scaling factor(s)
    """
    return (sub_s - truth_s).abs() / scaling
