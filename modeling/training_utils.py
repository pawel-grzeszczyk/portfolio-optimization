import torch

def train_step(model, X_train, Y_train, Y_train_true_labels, criterion, optimizer, profit_calc_fn):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)

    # Calculate loss
    loss = criterion(outputs, Y_train_true_labels)
    _, profit = profit_calc_fn(output_weights=outputs, Y=Y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return loss, profit

def test_step(model, X_test, Y_test, Y_test_true_labels, criterion, profit_calc_fn):
    model.eval()
    with torch.no_grad():
        # Forward pass
        test_outputs = model(X_test)

        # Calculate loss
        test_loss = criterion(test_outputs, Y_test_true_labels)
        _, test_profit = profit_calc_fn(output_weights=test_outputs, Y=Y_test)

    return test_loss, test_profit


def train(model: torch.nn.Module, 
          X_train,
          Y_train,
          Y_train_true_labels,
          X_test,
          Y_test,
          Y_test_true_labels,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          profit_calc_fn,
          epochs: int,
          print_every: int):
    
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        "train_profit": [],
        "test_loss": [],
        "test_profit": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        train_loss, train_profit = train_step(model=model,
                                              X_train=X_train,
                                              Y_train=Y_train,
                                              Y_train_true_labels=Y_train_true_labels,
                                              criterion=criterion,
                                              optimizer=optimizer,
                                              profit_calc_fn=profit_calc_fn)
        test_loss, test_profit = test_step(model=model,
                                        X_test=X_test,
                                        Y_test=Y_test,
                                        Y_test_true_labels=Y_test_true_labels,
                                        criterion=criterion,
                                        profit_calc_fn=profit_calc_fn)
        # 4. Print out what's happening
        if (epoch+1) % print_every == 0:
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_profit: {train_profit:.2f}% | "
                f"test_loss: {test_loss:.4f} | "
                f"test_profit: {test_profit:.2f}%"
            )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_profit"].append(train_profit.item() if isinstance(train_profit, torch.Tensor) else train_profit)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_profit"].append(test_profit.item() if isinstance(test_profit, torch.Tensor) else test_profit)

    # 6. Return the filled results at the end of the epochs
    return results