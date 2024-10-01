import pandas as pd

class CSVLoader:
    """
    Load CSV File 
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
    
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)