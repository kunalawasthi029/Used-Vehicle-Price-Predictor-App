"""
manufacturer_mapping.py - Curated real-world feature combinations per manufacturer.

This file contains hand-verified mappings of what vehicle types, fuel types,
cylinder counts, drive types, and transmissions each manufacturer actually
offers. This prevents unrealistic combinations (e.g., Kia bus, Porsche van)
from appearing in the Streamlit UI.

Source: Real-world manufacturer product lines (2015-2022 model years).
"""

# Each manufacturer maps to the feature values it realistically supports.
# Values must match exactly what exists in the training dataset's categories.

MANUFACTURER_FEATURES = {

    "acura": {
        "type": ["sedan", "SUV", "coupe"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic"],
    },

    "alfa-romeo": {
        "type": ["sedan", "SUV", "coupe"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "aston-martin": {
        "type": ["coupe", "convertible", "sedan"],
        "fuel": ["gas"],
        "cylinders": ["8 cylinders", "12 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "audi": {
        "type": ["sedan", "SUV", "coupe", "convertible", "wagon", "hatchback"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["4wd", "fwd"],
        "transmission": ["automatic", "manual"],
    },

    "bmw": {
        "type": ["sedan", "SUV", "coupe", "convertible", "wagon", "hatchback"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd", "fwd"],
        "transmission": ["automatic", "manual"],
    },

    "buick": {
        "type": ["sedan", "SUV"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic"],
    },

    "cadillac": {
        "type": ["sedan", "SUV", "coupe", "wagon"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd", "fwd"],
        "transmission": ["automatic"],
    },

    "chevrolet": {
        "type": ["sedan", "SUV", "coupe", "pickup", "truck", "convertible",
                 "hatchback", "van", "wagon"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["fwd", "rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "chrysler": {
        "type": ["sedan", "mini-van", "wagon"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic"],
    },

    "dodge": {
        "type": ["sedan", "SUV", "coupe", "mini-van", "wagon"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["fwd", "rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "ferrari": {
        "type": ["coupe", "convertible"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["8 cylinders", "12 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "fiat": {
        "type": ["hatchback", "sedan", "convertible"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders"],
        "drive": ["fwd"],
        "transmission": ["automatic", "manual"],
    },

    "ford": {
        "type": ["sedan", "SUV", "pickup", "truck", "coupe", "convertible",
                 "hatchback", "van", "wagon"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders", "10 cylinders"],
        "drive": ["fwd", "rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "gmc": {
        "type": ["SUV", "pickup", "truck", "van"],
        "fuel": ["gas", "diesel"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "harley-davidson": {
        "type": ["other"],
        "fuel": ["gas"],
        "cylinders": ["other"],
        "drive": ["rwd"],
        "transmission": ["manual", "other"],
    },

    "honda": {
        "type": ["sedan", "SUV", "coupe", "hatchback", "mini-van", "truck", "pickup"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "hyundai": {
        "type": ["sedan", "SUV", "coupe", "hatchback"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "infiniti": {
        "type": ["sedan", "SUV", "coupe", "convertible"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "jaguar": {
        "type": ["sedan", "SUV", "coupe", "convertible", "wagon"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "jeep": {
        "type": ["SUV", "pickup", "truck"],
        "fuel": ["gas", "diesel", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["4wd", "rwd", "fwd"],
        "transmission": ["automatic", "manual"],
    },

    "kia": {
        "type": ["sedan", "SUV", "hatchback", "mini-van", "wagon"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "land rover": {
        "type": ["SUV"],
        "fuel": ["gas", "diesel", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["4wd"],
        "transmission": ["automatic"],
    },

    "lexus": {
        "type": ["sedan", "SUV", "coupe", "convertible", "hatchback", "wagon"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd", "fwd"],
        "transmission": ["automatic"],
    },

    "lincoln": {
        "type": ["sedan", "SUV"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic"],
    },

    "mazda": {
        "type": ["sedan", "SUV", "hatchback", "convertible", "wagon"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "mercedes-benz": {
        "type": ["sedan", "SUV", "coupe", "convertible", "wagon", "hatchback", "van"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders", "12 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "mini": {
        "type": ["hatchback", "convertible", "coupe", "wagon"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["3 cylinders", "4 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "mitsubishi": {
        "type": ["sedan", "SUV", "hatchback", "coupe"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "nissan": {
        "type": ["sedan", "SUV", "coupe", "hatchback", "truck", "pickup",
                 "van", "convertible"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["fwd", "rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "porsche": {
        "type": ["sedan", "SUV", "coupe", "convertible", "hatchback", "wagon"],
        "fuel": ["gas", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "ram": {
        "type": ["pickup", "truck", "van"],
        "fuel": ["gas", "diesel"],
        "cylinders": ["6 cylinders", "8 cylinders"],
        "drive": ["rwd", "4wd"],
        "transmission": ["automatic"],
    },

    "rover": {
        "type": ["SUV"],
        "fuel": ["gas"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["4wd"],
        "transmission": ["automatic"],
    },

    "subaru": {
        "type": ["sedan", "SUV", "wagon", "hatchback"],
        "fuel": ["gas", "hybrid"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["4wd"],
        "transmission": ["automatic", "manual"],
    },

    "tesla": {
        "type": ["sedan", "SUV", "hatchback"],
        "fuel": ["electric"],
        "cylinders": ["other"],
        "drive": ["4wd", "rwd"],
        "transmission": ["automatic"],
    },

    "toyota": {
        "type": ["sedan", "SUV", "coupe", "pickup", "truck", "hatchback",
                 "mini-van", "van", "wagon"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
        "drive": ["fwd", "rwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "volkswagen": {
        "type": ["sedan", "SUV", "hatchback", "convertible", "wagon"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic", "manual"],
    },

    "volvo": {
        "type": ["sedan", "SUV", "wagon", "hatchback"],
        "fuel": ["gas", "diesel", "hybrid", "electric"],
        "cylinders": ["4 cylinders", "6 cylinders"],
        "drive": ["fwd", "4wd"],
        "transmission": ["automatic"],
    },
}

# Fallback for "Unknown" manufacturer or any not in the mapping
DEFAULT_FEATURES = {
    "type": ["sedan", "SUV", "coupe", "pickup", "truck", "hatchback",
             "convertible", "wagon", "van", "mini-van"],
    "fuel": ["gas", "diesel", "hybrid", "electric"],
    "cylinders": ["4 cylinders", "6 cylinders", "8 cylinders"],
    "drive": ["fwd", "rwd", "4wd"],
    "transmission": ["automatic", "manual"],
}


def get_manufacturer_options(manufacturer, field):
    """Return valid options for a given manufacturer and feature field.

    Args:
        manufacturer: Manufacturer name (lowercase, as in dataset).
        field: One of 'type', 'fuel', 'cylinders', 'drive', 'transmission'.

    Returns:
        list[str]: Valid option values for that manufacturer + field.
    """
    if manufacturer in MANUFACTURER_FEATURES:
        options = MANUFACTURER_FEATURES[manufacturer].get(field)
        if options:
            return options
    return DEFAULT_FEATURES.get(field, [])

