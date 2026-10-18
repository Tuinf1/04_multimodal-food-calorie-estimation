from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs=10,
    learning_rate=1e-3,
    patience=3,
    checkpoint_path="artifacts/best_model.pt",
):
    criterion = nn.L1Loss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()

        # Замороженные энкодеры оставляем в режиме оценки
        model.image_encoder.eval()
        model.text_encoder.eval()

        train_error = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"Эпоха {epoch + 1}/{epochs}",
        ):
            images = batch["image"].to(device)
            texts = batch["text"]
            masses = batch["mass"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                predictions = model(
                    images=images,
                    texts=texts,
                    masses=masses,
                )

                loss = criterion(
                    predictions,
                    targets,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_error += (
                loss.item() * targets.size(0)
            )

        train_mae = (
            train_error
            / len(train_loader.dataset)
        )

        model.eval()
        validation_error = 0.0

        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["image"].to(device)
                texts = batch["text"]
                masses = batch["mass"].to(device)
                targets = batch["target"].to(device)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    predictions = model(
                        images=images,
                        texts=texts,
                        masses=masses,
                    )

                validation_error += (
                    torch.abs(predictions - targets)
                    .sum()
                    .item()
                )

        validation_mae = (
            validation_error
            / len(validation_loader.dataset)
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_mae": train_mae,
                "validation_mae": validation_mae,
            }
        )

        print(
            f"Эпоха {epoch + 1}: "
            f"train MAE = {train_mae:.2f}, "
            f"validation MAE = {validation_mae:.2f}"
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "validation_mae": validation_mae,
                },
                checkpoint_path,
            )

            print(
                "Сохранена лучшая модель: "
                f"MAE = {validation_mae:.2f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Обучение остановлено по early stopping")
            break

    return history