import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern} \usepackage[T1]{fontenc}",
})

TITLE_SIZE  = 30
LABEL_SIZE  = 22
TICK_SIZE   = 22
LEGEND_SIZE = 20
LINE_WIDTH  = 3