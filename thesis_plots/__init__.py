TEXT_WIDTH_INCH = 6.22404097223
# default pyplot figure size w=6.4, h=4.8


def textwidth_to_figsize(w_frac, aspect=2/3):
    w = w_frac * TEXT_WIDTH_INCH
    h = w * aspect
    return w, h
