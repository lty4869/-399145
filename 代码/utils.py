import pandas as pd


def to_timestamp_ignore_errors(x):
    try:
        return pd.to_datetime(x).timestamp()
    except Exception:
        return x

def print_lr_formula(lr, ):
    sl = []
    for i, j in enumerate(lr.coef_[2:]):
        if j > 0.001 or j < -0.001:
            sl.append('{:.2f}log^{}(x)'.format(j, i + 2))
        else:
            ts = "{:.2e}".format(j)
            ts = ts.split('e')
            sl.append("{}^{}log^{}(x)".format(ts[0], "{" + ts[1] + "}", i + 2))
    s = '$y=e^{' + \
        '{:.2f}'.format(lr.intercept_) + \
        '+{:.2f}log(x)+'.format(lr.coef_[1]) + \
        "+".join(sl) + \
        '}$'
    print(s.replace('+-', '-'))