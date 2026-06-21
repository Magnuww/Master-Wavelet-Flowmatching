
def r2(x, y):
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - x) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score
