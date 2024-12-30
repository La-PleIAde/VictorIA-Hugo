from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def feature_name_combiner(feature, category):
    return str(category)


class CanonData:
    """
    Load the metadata corresponding to the *canon corpus*. It also creates
    one new variable "period" which correspond to the period at which the
    document was published, using the publication date.

    :param path: Local path to the csv
    :param ids: List of doc ids to keep, for example:
    ["filename_1", "filename_2", ..., "filename_n"]
    """

    def __init__(self, path, ids: Optional[List[str]] = None):
        self.data = pd.read_csv(path, index_col=0)
        self.cat_variables = ["author", "gender", "period"]
        # Date segments
        self.period_segments = [
            [1800, 1826],
            [1827, 1850],
            [1851, 1869],
            [1870, 1899],
            [1900, 1945],
            [1946, 2024],
        ]
        self.periods = pd.DataFrame(
            self.period_segments, columns=["start", "end"]
        )
        # Add period column
        self.data["period"] = self.data["date_publication"].apply(
            lambda x: self.get_seg_from_date(x)
        )
        self.data.doc_id = self.data.doc_id.str.replace(".csv", "")
        self.data.set_index("doc_id", inplace=True)
        if ids is not None:
            assert all([i in self.data.index for i in ids])
            self.data = self.data.loc[ids]

        # Handle missing values
        self.data.loc[self.data["author"].isna(), "author"] = "unknown"
        self.data.loc[self.data["gender"].isna(), "gender"] = "unknown"

        # Handle types
        self.data[["author", "gender", "label", "period"]] = self.data[
            ["author", "gender", "label", "period"]
        ].astype("category")

        # Make sure that label is correctly defined
        assert set(self.data.label.unique()) == set(["non-canon", "canon"])
        # Make sure canon is encoded as 1
        self.data.label = self.data.label.cat.set_categories(
            ["non-canon", "canon"]
        )
        # Assert no NaN
        assert not self.data.isna().any().any(), "NaNs in metadata!"

    def get_seg_from_date(self, date: int) -> str:
        if date != np.nan:
            mask = np.argmax(
                (self.periods["start"] <= date) & (date <= self.periods["end"])
            )
            return "-".join(map(str, self.periods.loc[mask, :]))

    def encode_categories(
        self, handle_unknown="ignore"
    ) -> Union[pd.DataFrame, OneHotEncoder]:
        categories = [
            self.data[c].cat.categories.tolist() for c in self.cat_variables
        ]
        if handle_unknown == "ignore":
            # Remove the category "unknown", the feature will be encoded has
            # a 0 vector for these individuals
            categories = [
                [c for c in cat if c != "unknown"] for cat in categories
            ]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            feature_name_combiner=feature_name_combiner,
        )
        encoder.fit(self.data[self.cat_variables])
        encoded = encoder.transform(self.data[self.cat_variables])
        encoded = pd.DataFrame(
            encoded.toarray(),
            index=self.data.index,
            columns=encoder.get_feature_names_out(),
        )
        return encoded, encoder
