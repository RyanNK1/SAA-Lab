"""Construction of the commodities sleeve from its two components.

Commodities is one bucket in the allocation. A slider sets its internal split
between gold and commodities ex-gold, anywhere from 100/0 to 0/100.

The sleeve is built *before* the portfolio is measured, so a five-asset panel
goes into the optimizer rather than a six-asset one. This matters more than it
sounds: the sleeve's return, volatility and every one of its correlations
change as the slider moves. It is not a fixed asset with a fixed character.

The two components behave very differently. Gold rose during the financial
crisis; broad commodities fell about 25% in a single month. So a gold-heavy
sleeve reads as a diversifier while an oil-heavy one reads as equity risk, and
the same nominal allocation to "commodities" can mean opposite things. That
interaction is worth surfacing rather than hiding behind one label.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.panels import ReturnPanel

GOLD = "gold"
COMMODITIES_EX_GOLD = "commodities_ex_gold"
SLEEVE = "commodities"

DEFAULT_GOLD_WEIGHT = 0.5


@dataclass(frozen=True)
class SleeveSpec:
    """How the commodities sleeve is composed.

    `gold_weight` is the slider position: 1.0 is entirely gold, 0.0 entirely
    commodities ex-gold. Stored as a fraction rather than a percentage so it
    can be used directly in arithmetic without a conversion step that someone
    will eventually forget.
    """

    gold_weight: float = DEFAULT_GOLD_WEIGHT

    def __post_init__(self) -> None:
        if not isinstance(self.gold_weight, (int, float)):
            raise TypeError("gold_weight must be a number")
        if not 0.0 <= self.gold_weight <= 1.0:
            raise ValueError(
                f"gold_weight must be between 0.0 and 1.0, got {self.gold_weight}. "
                f"The slider spans 100% gold to 100% commodities ex-gold."
            )

    @property
    def ex_gold_weight(self) -> float:
        return 1.0 - self.gold_weight

    @property
    def is_pure_gold(self) -> bool:
        return self.gold_weight == 1.0

    @property
    def is_pure_ex_gold(self) -> bool:
        return self.gold_weight == 0.0

    def describe(self) -> str:
        return f"{self.gold_weight:.0%} gold / {self.ex_gold_weight:.0%} ex-gold"


def build_sleeve(
    panel: ReturnPanel,
    spec: SleeveSpec | float = DEFAULT_GOLD_WEIGHT,
    sleeve_name: str = SLEEVE,
) -> ReturnPanel:
    """Combine gold and commodities ex-gold into a single sleeve.

    Returns a new panel with both components replaced by one column. Column
    order is otherwise preserved, with the sleeve taking the position of the
    first component it replaces, so downstream output stays stable as the
    slider moves.

    The sleeve is rebalanced every period, which is what makes its return the
    weighted average of its components' returns. Holding it un-rebalanced would
    let the composition drift away from the slider's stated position -- so the
    slider would stop meaning what it says, which is the one thing it must not
    do.

    A panel without both components is returned unchanged rather than raising:
    a user who deselected commodities entirely has no sleeve to build, and that
    is a legitimate request, not an error.
    """
    if isinstance(spec, (int, float)):
        spec = SleeveSpec(float(spec))

    has_gold = GOLD in panel.assets
    has_ex_gold = COMMODITIES_EX_GOLD in panel.assets

    if not has_gold and not has_ex_gold:
        return panel

    if has_gold != has_ex_gold:
        # Only one component present. Rename it to the sleeve so downstream
        # code sees a consistent asset name, but do not silently pretend the
        # slider was honoured -- it cannot be with one component.
        present = GOLD if has_gold else COMMODITIES_EX_GOLD
        renamed = panel.returns.rename(columns={present: sleeve_name})
        return ReturnPanel(renamed, panel.periods_per_year)

    if sleeve_name in panel.assets:
        raise ValueError(
            f"Panel already has a column named {sleeve_name!r}; cannot build "
            f"the sleeve into it"
        )

    combined = (
        panel.returns[GOLD] * spec.gold_weight
        + panel.returns[COMMODITIES_EX_GOLD] * spec.ex_gold_weight
    )

    out = panel.returns.copy()
    position = out.columns.get_loc(GOLD)
    out = out.drop(columns=[GOLD, COMMODITIES_EX_GOLD])
    out.insert(min(position, len(out.columns)), sleeve_name, combined)

    return ReturnPanel(out, panel.periods_per_year)


def sleeve_components(panel: ReturnPanel) -> list[str]:
    """Which sleeve components are present in a panel. Useful for the UI."""
    return [a for a in (GOLD, COMMODITIES_EX_GOLD) if a in panel.assets]


def sleeve_sensitivity(
    panel: ReturnPanel, steps: int = 11
) -> pd.DataFrame:
    """How the sleeve's own statistics change as the slider moves.

    One row per slider position, with the sleeve's annualised return and
    volatility, and its correlation with each remaining asset. Makes the point
    that "commodities" is not one thing -- the same allocation label can be a
    diversifier or a second helping of equity risk depending on this setting.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if GOLD not in panel.assets or COMMODITIES_EX_GOLD not in panel.assets:
        raise ValueError(
            f"Sensitivity needs both {GOLD} and {COMMODITIES_EX_GOLD} in the panel"
        )

    others = [a for a in panel.assets if a not in (GOLD, COMMODITIES_EX_GOLD)]

    rows = []
    for i in range(steps):
        gold_weight = i / (steps - 1)
        built = build_sleeve(panel, gold_weight)

        row = {
            "gold_weight": gold_weight,
            "ann_return": built.ann_return()[SLEEVE],
            "ann_vol": built.ann_vol()[SLEEVE],
        }
        corr = built.corr()
        for other in others:
            row[f"corr_{other}"] = corr.loc[SLEEVE, other]
        rows.append(row)

    return pd.DataFrame(rows)
