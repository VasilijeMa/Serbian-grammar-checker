import torch
import torch.optim as optim
import torch.nn as nn

def train(model, num_epochs, dataloader, checkpoint_path):
    print("Training new model.")
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}...")
        epoch_loss = 0.0
        for inputs_before, inputs_after, targets in dataloader:
            outputs = model(inputs_before, inputs_after)

            if outputs.shape != targets.shape:
                raise ValueError(f"Output shape {outputs.shape} does not match target shape {targets.shape}")
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    state = {
        'epoch' : num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path.format(epoch=num_epochs))