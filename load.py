import pandas as pd


def load_data(path: str, format: str = 'file',**kwargs) -> pd.DataFrame:
    """Read data from file or database."""
    if format == 'file':
        file_ext = path.split(".")[-1].lower()
        if file_ext == 'json':
            df = pd.read_json(path)
        elif file_ext == 'csv':
            df = pd.read_csv(path)
        else:
            raise NotImplementedError(f"File extension {file_ext} is not supported yet")
    elif format == 'db':
        raise NotImplementedError("Database support is not implemented yet")
    else:
        raise ValueError(f"Unsupported data source format: {format}")

    return df




if __name__ == "__main__":
    df = load_data("data/data.json")
    print(df.head())
