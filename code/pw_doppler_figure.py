
import numpy as np
from latexify import *

fc = 8e6    # 8Mhz
N_cycles = 12
PRF = 20e3  # 20kHz
num_pulses = 2
dt = 1 / (40 * fc)  # Sampling interval

figures_path = 'figures/theory'
filename = 'emitted_waveform.pdf'
WAVEFORM_COLOR = '#D4772B'  # calm burnt orange


def plot_emitted_waveform():
    T_PRF = 1 / PRF  # Pulse Repetition Interval
    T = N_cycles / fc  # Duration of one pulse

    pad = 0
    t = np.arange(-2e-6, num_pulses * T_PRF, dt)
    waveform = np.zeros(len(t))

    burst_starts = []
    for i in range(num_pulses):
        t_start = pad + i * T_PRF
        burst_starts.append(t_start)
        mask = (t >= t_start) & (t < t_start + T)
        waveform[mask] = np.sign(np.sin(2 * np.pi * fc * (t[mask] - t_start)))

    t_us = t * 1e6
    view_pad_left  = 0.75  # µs before each burst
    view_pad_right = 0.75  # µs after each burst

    fig, axes = plt.subplots(1, num_pulses, sharey=True, figsize=(16, 9),
                             gridspec_kw={'wspace': 0})

    S = 0.5   # snap step in µs; 0.5 makes burst centre == panel centre
    tick_step = 1.0  # displayed tick spacing in µs
    for ax, bs in zip(axes, burst_starts):
        bs_us = bs * 1e6
        x_left  = bs_us - view_pad_left
        x_right = bs_us + T * 1e6 + view_pad_right
        xlim_min = (np.floor(x_left  / S - 0.5) + 0.5) * S
        xlim_max = (np.ceil( x_right / S - 0.5) + 0.5) * S
        mask_plot = t_us <= 52
        ax.plot(t_us[mask_plot], waveform[mask_plot], color=WAVEFORM_COLOR, linewidth=LINE_WIDTH)
        ax.set_xlim(xlim_min, xlim_max)
        ax.xaxis.set_major_locator(plt.MultipleLocator(tick_step))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid()
    axes[-1].set_xlim(axes[-1].get_xlim()[0], 52.2)

    # Hide inner spines at the break; slanted cut mark on bottom only
    d = 0.015
    for i, ax in enumerate(axes):
        ax.tick_params(axis='y', left=False, labelleft=False)
        if i < num_pulses - 1:
            ax.spines['right'].set_visible(False)
            kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=2)
            ax.plot((1 - d, 1 + d), (-d, +d), **kw)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelsize=TICK_SIZE)
            kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=2)
            ax.plot((-d, +d), (-d, +d), **kw)

    # Annotate first burst duration with a dimension bar
    ax0 = axes[0]
    T_us = T * 1e6
    y_bar = 1.15
    cap = 0.05  # half-height of vertical end caps
    ax0.plot([0, T_us], [y_bar, y_bar], 'k-', lw=1.5)
    ax0.plot([0, 0],     [y_bar - cap, y_bar + cap], 'k-', lw=1.5)
    ax0.plot([T_us, T_us], [y_bar - cap, y_bar + cap], 'k-', lw=1.5)
    ax0.text(T_us / 2, y_bar + 0.08,
             r'$T = N_{cycles} / f_c$', ha='center', va='bottom',
             fontsize=LABEL_SIZE)

    # Annotate one carrier period on the second burst
    ax1 = axes[1]
    bs1_us = burst_starts[1] * 1e6
    period_us = 1 / fc * 1e6
    x_p1 = bs1_us        # first peak
    x_p2 = bs1_us + period_us  # second peak
    ax1.plot([x_p1, x_p2], [y_bar, y_bar], 'k-', lw=1.5)
    ax1.plot([x_p1, x_p1], [y_bar - cap, y_bar + cap], 'k-', lw=1.5)
    ax1.plot([x_p2, x_p2], [y_bar - cap, y_bar + cap], 'k-', lw=1.5)
    ax1.text((x_p1 + x_p2) / 2, y_bar + 0.08,
             r'$1 / f_c$', ha='center', va='bottom', fontsize=LABEL_SIZE)

    axes[0].set_ylim(-1.5, 1.5)  # expand to fit both annotations

    # T_PRF bar end caps in each panel's data coords
    T_PRF_us = T_PRF * 1e6
    y_bar2 = -1.15
    axes[0].plot([0, 0], [y_bar2 - cap, y_bar2 + cap], 'k-', lw=1.5, clip_on=False)
    axes[1].plot([T_PRF_us, T_PRF_us], [y_bar2 - cap, y_bar2 + cap], 'k-', lw=1.5, clip_on=False)

    # Place tiny circles at the break and near the right edge of the last panel
    fig.canvas.draw()  # finalise transforms after ylim is set
    for i in range(num_pulses - 1):
        x_fig = axes[i].get_position().x1
        pt_y = fig.transFigure.inverted().transform(axes[i].transData.transform((0, 0)))
        fig.text(x_fig, pt_y[1], r'$\circ\;\circ\;\circ$',
                 ha='center', va='center', fontsize=10, backgroundcolor='white')
    # Trailing dots at x=52µs (right end of last panel)
    pt_trail = fig.transFigure.inverted().transform(axes[-1].transData.transform((52.165, 0)))
    fig.text(pt_trail[0], pt_trail[1], r'$\circ\;\circ\;\circ$',
             ha='right', va='center', fontsize=10, backgroundcolor='white')

    # T_PRF bar: horizontal line spanning both panels, drawn in figure coords
    from matplotlib.lines import Line2D
    pt0 = fig.transFigure.inverted().transform(axes[0].transData.transform((0, y_bar2)))
    pt1 = fig.transFigure.inverted().transform(axes[1].transData.transform((T_PRF_us, y_bar2)))
    fig.add_artist(Line2D([pt0[0], pt1[0]], [pt0[1], pt1[1]],
                          transform=fig.transFigure, color='k', lw=1.5, clip_on=False))
    fig.text((pt0[0] + pt1[0]) / 2, pt0[1] - 0.03,
             r'$T_{PRF} = 1 / PRF$', ha='center', va='top', fontsize=LABEL_SIZE)

    fig.supxlabel('Time ($\\mu$s)', fontsize=LABEL_SIZE, y=0.04)
    axes[0].set_ylabel('Amplitude', fontsize=LABEL_SIZE, labelpad=20)
    fig.suptitle('Ideal Electrical Waveform for PW Doppler', fontsize=TITLE_SIZE, y=0.95)

    plt.savefig(f'{figures_path}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot_emitted_waveform()