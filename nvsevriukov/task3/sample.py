from modules import MLPImputer

import torch

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader

N_COLUMNS = 5
TRAIN_SIZE = 10000

BATCH_SIZE = 512
NUM_EPOCHS = 1000

train_mask = torch.randint(0, 2, size=(TRAIN_SIZE, N_COLUMNS), dtype=torch.bool)
train_data = torch.randn(TRAIN_SIZE, N_COLUMNS) * train_mask

train_dataset = TensorDataset(train_data, train_mask)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MLPImputer(
    n_columns=N_COLUMNS,
    hidden_embed_dim=10,
    output_embed_dim=10,
    bottleneck_output_dim=10,
    bottleneck_layers=2,
    head_layers=2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

pbar = tqdm(range(NUM_EPOCHS))
for _ in pbar:

    for x, m in train_loader:
        x, m, d = MLPImputer.get_dropped_mask(x, m, drop_rate=0.3)

        optimizer.zero_grad()
        output = model.forward(x, m)
        loss = model.loss(output, x, d)
        loss.backward()

        optimizer.step()
