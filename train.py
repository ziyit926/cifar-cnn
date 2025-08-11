model = None
loss_fn = None
optimizer = None
device = None
train_loader = None


def train():
    global model, loss_fn, optimizer, device, train_loader


    total_loss = 0 # sum of batch losses
    total_correct = 0 # sum of correct predictions across all samples
    total_samples = len(train_loader.dataset) # how many examples in the whole dataset

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move batch to the desired device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        # Predict outputs for this batch and calculate how far predictions are from true labels
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backpropagation
        optimizer.zero_grad() # clear previously stored gradients
        loss.backward() # compute gradients for this batch
        optimizer.step() # update model parameters using the gradients

        # Track running totals and calculate how many predictions were correct in this batch
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

        if batch_idx % 100 == 0:
            current = (batch_idx + 1) * len(inputs)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{total_samples:>5d}]")

    # Summary of the train
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100
    print(f"\nTrain accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}\n")
