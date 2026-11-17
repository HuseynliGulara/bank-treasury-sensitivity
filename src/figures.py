"""Figure 1, Figure 2 and Appendix Figure A1."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg
from .econometrics import rolling_beta, rolling_betas
from .tables import exclude_covid

GROUPS = {"Commercial banks": cfg.COMMERCIAL,
          "Investment banks": cfg.INVESTMENT,
          "Regional banks (KRE)": ["KRE"]}


def _save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi=200)
    fig.savefig(cfg.FIGURES_DIR / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _event_lines(ax):
    ax.axvline(pd.Timestamp(cfg.BREAK_DATE), ls="--", c="grey", lw=0.8)
    ax.axvline(pd.Timestamp(cfg.SVB_DATE), ls=":", c="grey", lw=0.8)
    ax.axhline(0, c="black", lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def _group_panels(roll, market=None):
    """Three panels (commercial, investment, regional), optionally with the market line."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, sharey=True)
    for ax, (title, members) in zip(axes, GROUPS.items()):
        for b in members:
            ax.plot(roll.index, roll[b], lw=1.1, label=b)
        if market is not None:
            ax.plot(market.index, market, lw=1.2, color="black", ls="--", label="S&P 500 TR")
        _event_lines(ax)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left",
                  ncol=1 if market is None else 2)
    axes[1].set_ylabel("252-day rolling beta on ΔDGS10")
    return fig, axes


def figure1(sample):
    """252-day rolling beta on the 10-year yield, Model 1."""
    fig, axes = _group_panels(rolling_betas(sample, cfg.X_MODEL1))
    axes[0].text(pd.Timestamp(cfg.BREAK_DATE), axes[0].get_ylim()[1] * 0.9,
                 " Mar 2022", fontsize=8, color="grey")
    axes[0].text(pd.Timestamp(cfg.SVB_DATE), axes[0].get_ylim()[1] * 0.75,
                 " SVB", fontsize=8, color="grey")
    fig.tight_layout()
    _save(fig, "figure1_rolling_beta_model1")


def figure2(sample):
    """Rolling betas for banks and the S&P 500, all from Model 2."""
    market = rolling_beta(sample, f"ret_{cfg.MARKET}", cfg.X_MODEL2)
    fig, _ = _group_panels(rolling_betas(sample, cfg.X_MODEL2), market)
    fig.tight_layout()
    _save(fig, "figure2_rolling_beta_model2_with_market")


def figureA1(sample):
    """Full sample vs excluding 2 March - 30 April 2020."""
    ex = exclude_covid(sample)
    panels = [(b, cfg.X_MODEL1) for b in cfg.BANKS] + [(cfg.MARKET, cfg.X_MODEL2)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.0 * len(panels)),
                             sharex=True, sharey=True)
    for ax, (a, X) in zip(axes, panels):
        full = rolling_beta(sample, f"ret_{a}", X)
        excl = rolling_beta(ex, f"ret_{a}", X)
        ax.plot(full.index, full, lw=1.0, label="Full sample")
        ax.plot(excl.index, excl, lw=1.0, ls="--", label="Excl. Mar-Apr 2020")
        ax.axvline(pd.Timestamp(cfg.BREAK_DATE), ls=":", c="grey", lw=0.8)
        ax.axhline(0, c="black", lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(a + (" (no market control)" if a == cfg.MARKET else ""),
                     fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "figureA1_rolling_beta_ex_covid")
