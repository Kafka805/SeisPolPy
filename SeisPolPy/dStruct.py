"""
@author: Austin Abreu
"""

import numpy as np
import pandas as pd


class dataStruct:
    """
    Wrapper for data structure creation utilizing pandas DataFrames. Designed
    for use with SeisPol.

    Initalizing Variables:
        headers (list[str]): A list containing column names.
        length (int): Integer number describing how many rows to initialize.
    """

    def __init__(self, headers: list[str] = None, rows: int = 1):
        if headers is None:
            headers = [
                "Rectilinearity",
                "Planarity",
                "Azimuth",
                "Incident",
                "Normalized Diff",
            ]
        self.headers = headers
        self.length = rows

        # The goal of the wrapper is to be multi-modal based on the number of rows.
        # If the user wants a single collection of labelled values, then "length"
        if rows == 1:
            self.body = pd.Series(index=headers)
        else:
            self.body = pd.DataFrame(columns=headers, index=range(rows))

    def __str__(self) -> str:
        return f"{self.body}"


def main():
    return print("dStruct: Why are you running this module?")


if __name__ == "__main__":
    main()
