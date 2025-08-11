import torch

model = None
loss_fn = None
device = None
test_loader = None

def test():
    global model, loss_fn, device, test_loader

    model.eval()

    total_samples = len(test_loader.dataset)  # Total number of examples in dataset
    total_batches = len(test_loader)          # How many batches we will run through

    total_loss = 0.0     # To add up losses for all batches
    total_correct = 0    # To count all correct predictions

    # Turn off gradient calculations to save memory and computations
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the right device (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model predictions for this batch
            outputs = model(inputs)

            # Calculate loss for this batch and add to total loss
            batch_loss = loss_fn(outputs, labels).item()
            total_loss += batch_loss

            # Find predicted classes (the class with highest score)
            predicted = outputs.argmax(dim=1)

            # Count how many predictions are correct in this batch and add to total
            total_correct += (predicted == labels).sum().item()

    # Calculate average loss per batch
    average_loss = total_loss / total_batches

    # Calculate accuracy as percent of correctly predicted samples
    accuracy = (total_correct / total_samples) * 100

    # Print the results nicely formatted
    print(f"Test accuracy: {accuracy:.4f}% | Average loss: {average_loss:.4f}\n")
