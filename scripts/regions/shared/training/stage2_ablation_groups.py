"""Feature group definitions for Stage 2 ablation."""

ABLATION_GROUPS: dict[str, list[str]] = {
    "terrain": ["elevation", "slope", "WorldClim"],
    "biodiversity": ["GSN"],
    "deforestation": ["deforestation", "wildfire", "NDVI", "landcover"],
    "pa_momentum": ["pa_momentum"],
    "infrastructure": ["dist_", "HNTL", "GPW", "powerplants", "economic_value"],
    "policy": ["policy"],
}
