import torch
from torch.utils.data import Dataset
from kusa import DatasetClient, DatasetSDKException

class RemoteDataset(Dataset):
    def __init__(self, client: DatasetClient, batch_size: int):
        """
        Initializes the RemoteDataset.

        Args:
            client (DatasetClient): An instance of DatasetClient.
            batch_size (int): Number of samples per batch.
        """
        self.client = client
        self.batch_size = batch_size
        self.init_data = self.client.initialize()
        self.total_rows = self.init_data["totalRows"]
        self.current_batch = 1
        self.max_batches = (self.total_rows + batch_size - 1) // batch_size  # Ceiling division

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        """
        Fetches a single data sample.

        Since data is fetched in batches, this method ensures the required batch is loaded.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: (input, label)
        """
        batch_number = (idx // self.batch_size) + 1
        if batch_number != self.current_batch:
            try:
                batch_data = self.client.fetch_batch(self.batch_size, batch_number)
                self.current_batch = batch_number
                # Assuming the last column is the label
                self.inputs = torch.tensor(batch_data.iloc[:, :-1].values, dtype=torch.float32)
                self.labels = torch.tensor(batch_data.iloc[:, -1].values, dtype=torch.long)
            except DatasetSDKException as e:
                raise RuntimeError(f"Failed to fetch batch {batch_number}: {e}")

        sample_idx = idx % self.batch_size
        if sample_idx >= len(self.inputs):
            raise IndexError("Index out of range in the current batch.")
        
        return self.inputs[sample_idx], self.labels[sample_idx]
