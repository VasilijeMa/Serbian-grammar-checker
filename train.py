import torch
import torch.optim as optim
import torch.nn as nn

def train(model, num_epochs, dataloader, checkpoint_path):
    print("Training new model.")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    i = 0
    original_loss = []
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}...")
        epoch_loss = 0.0
        i += 1
        for inputs_before, inputs_after, targets in dataloader:
            # print(f"Before {inputs_before}")
            # print(f"Target {targets}")
            # print(f"After {inputs_after}")
            outputs = model(inputs_before, inputs_after)

            if outputs.shape != targets.shape:
                raise ValueError(f"Output shape {outputs.shape} does not match target shape {targets.shape}")
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print(f"LOSS {loss.item()}")
            if i == 1: original_loss.append(loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    print(f"Original loss {original_loss}")

    # state = {
    #     'epoch' : num_epochs,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }
    # torch.save(state, checkpoint_path.format(epoch=num_epochs))