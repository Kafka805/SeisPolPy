# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:43:36 2024

@author: Austin Abreu
"""
import numpy as np
import pandas as pd
    
class dataStruct:
    def __init__(self, headers: list[str] = None, 
                 length: int = None):
        if headers is None:
            headers = [
                        'Rectilinearity',
                        'Planarity',
                        'Azimuth',
                        'Incident',
                        'Normalized Diff'
                        ]
        self.headers = headers
        
        if length is None:
            length = 1
        self.length = length
        
        if length == 1:
            self.body = pd.Series(dict(headers))
        self.body = pd.DataFrame(columns=headers, index=range(length))
        
    def __str__(self) -> str:
        return f"{self.body}"
    
        
def main():
    pass
    
if __name__ == '__main__':
    main()
