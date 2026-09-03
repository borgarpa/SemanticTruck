"""Which acquisition campaign a truckload belongs to, and what it was carried on.

Both are read from the year at the end of the load name: `Truck_7_2016` is a 2016 load. The two
campaigns differ in more than date -- 2016 was semitrailers carrying a single stack, 2021 was
B-doubles carrying two -- which is what decides whether a load is split into bays at all.

**No imports.** Both halves of the package need this, and the training stage runs in an environment
that has neither pandas nor Open3D.
"""

# Campaign year -> the truck configuration it used. 2016 was semitrailers, carrying one stack;
# 2021 was B-doubles, carrying two.
CONFIG = {'2016': 'Semitrailer', '2021': 'B-double'}

CAMPAIGN_YEARS = tuple(CONFIG)


def campaign_of(load_id: str) -> str:
    """The acquisition year of a load, from the end of its name."""
    return load_id.rsplit('_', 1)[-1]


def config_of(load_id: str) -> str:
    """Truck configuration, which the campaign determines."""
    return CONFIG.get(campaign_of(load_id), '')
