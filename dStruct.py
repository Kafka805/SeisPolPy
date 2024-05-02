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

    def __init__(self, headers: list[str] = None, length: int = None):
        if headers is None:
            headers = [
                "Rectilinearity",
                "Planarity",
                "Azimuth",
                "Incident",
                "Normalized Diff",
            ]
        self.headers = headers

        if length is None:
            length = 1
        self.length = length

        if length == 1:
            self.body = pd.Series(index=headers)
        else:
            self.body = pd.DataFrame(columns=headers, index=range(length))

    def __str__(self) -> str:
        return f"{self.body}"


def main():
    return print("dStruct: Why are you running this module?")


if __name__ == "__main__":
    main()
