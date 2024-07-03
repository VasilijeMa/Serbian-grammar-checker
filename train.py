import torch

def train(model, criterion, optimizer, NUM_EPOCHS, dataloader, checkpoint_path):
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}")
        i = 1
        epoch_loss = 0.0
        for input_before, input_after, target in dataloader:
            input_before = input_before.unsqueeze(1)
            input_after = input_after.unsqueeze(1)

            outputs = model(input_before, input_after)
            loss = criterion(outputs, target)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    state = {
        'epoch' : NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path.format(epoch=NUM_EPOCHS))