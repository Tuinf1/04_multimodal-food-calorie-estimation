import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from scripts.dataset import (
    create_dataloaders,
    load_and_prepare_data,
)
from scripts.model import CalorieModel


def set_seed(seed):
    """
    Фиксирует генераторы случайных чисел.

    Это позволяет воспроизводить разделение данных
    и результаты обучения.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    stage,
    statistics,
):
    """
    Обучает модель на одном этапе.

    После каждой эпохи рассчитывает train MAE
    и validation MAE. Лучшее состояние модели
    сохраняется в отдельный checkpoint.
    """

    # MAE используется как функция потерь
    criterion = nn.L1Loss()

    # В optimizer передаются только размороженные параметры
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=stage["learning_rate"],
    )

    # Mixed precision включается только при работе на GPU
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # Создаём папку для сохранения checkpoint
    checkpoint_path = Path(
        stage["checkpoint_path"]
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Считаем количество обучаемых параметров
    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        "Обучаемых параметров:",
        f"{trainable_count:,}",
    )

    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(stage["epochs"]):
        # Включаем режим обучения
        model.train()

        # Энкодеры работают без изменения статистик BatchNorm
        model.image_encoder.eval()
        model.text_encoder.eval()

        train_error = 0.0

        # Обучение на train
        for batch in tqdm(
            train_loader,
            desc=(
                f"{stage['name']}: "
                f"эпоха {epoch + 1}/{stage['epochs']}"
            ),
        ):
            images = batch["image"].to(device)
            texts = batch["text"]
            masses = batch["mass"].to(device)
            targets = batch["target"].to(device)

            # Очищаем старые градиенты
            optimizer.zero_grad(set_to_none=True)

            # Выполняем прямой проход
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

            # Вычисляем градиенты и обновляем веса
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_error += (
                loss.item() * targets.size(0)
            )

        # Рассчитываем средний train MAE
        train_mae = (
            train_error
            / len(train_loader.dataset)
        )

        # Переводим модель в режим оценки
        model.eval()
        validation_error = 0.0

        # Проверяем модель на validation
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

        # Рассчитываем средний validation MAE
        validation_mae = (
            validation_error
            / len(validation_loader.dataset)
        )

        # Добавляем результаты эпохи в историю
        history.append(
            {
                "stage": stage["name"],
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

        # Сохраняем модель при улучшении validation MAE
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            epochs_without_improvement = 0

            torch.save(
                {
                    "stage": stage["name"],
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "validation_mae": validation_mae,
                    "mass_mean": statistics["mass_mean"],
                    "mass_std": statistics["mass_std"],
                },
                checkpoint_path,
            )

            print(
                "Сохранена лучшая модель: "
                f"MAE = {validation_mae:.2f}"
            )
        else:
            epochs_without_improvement += 1

        # Проверяем условие ранней остановки
        if (
            epochs_without_improvement
            >= stage["patience"]
        ):
            print(
                "Обучение этапа остановлено "
                "по early stopping"
            )
            break

    return history


def train(config_path):
    """
    Выполняет полный пайплайн обучения.

    Загружает конфигурацию и данные, создаёт DataLoader
    и последовательно выполняет все включённые этапы.
    """

    # Загружаем YAML-конфигурацию
    with open(
        config_path,
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    # Фиксируем случайность
    set_seed(config["seed"])

    # Выбираем устройство
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Загружаем и подготавливаем данные
    dataframe = load_and_prepare_data(
        data_dir=config["data_dir"],
    )

    # Создаём DataLoader для всех выборок
    loaders, dataframes, statistics = create_dataloaders(
        dataframe=dataframe,
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        validation_size=config["validation_size"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    print("Устройство:", device)
    print("Train:", len(dataframes["train"]))
    print("Validation:", len(dataframes["validation"]))
    print("Test:", len(dataframes["test"]))

    full_history = []

    # Последовательно выполняем этапы из config.yaml
    for stage in config["stages"]:
        # Пропускаем отключённые этапы
        if not stage["enabled"]:
            print(
                f"Этап пропущен: {stage['name']}"
            )
            continue

        print()
        print("=" * 50)
        print("Этап:", stage["name"])
        print("=" * 50)

        # Создаём модель с настройками текущего этапа
        model = CalorieModel(
            fine_tune_image=stage["fine_tune_image"],
        ).to(device)

        initial_checkpoint = stage.get(
            "initial_checkpoint"
        )

        # Загружаем веса предыдущего этапа
        if initial_checkpoint is not None:
            initial_checkpoint = Path(
                initial_checkpoint
            )

            if not initial_checkpoint.is_file():
                raise FileNotFoundError(
                    "Не найден checkpoint: "
                    f"{initial_checkpoint}"
                )

            checkpoint = torch.load(
                initial_checkpoint,
                map_location=device,
                weights_only=True,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print(
                "Загружены начальные веса:",
                initial_checkpoint,
            )

        # Обучаем модель на текущем этапе
        stage_history = train_model(
            model=model,
            train_loader=loaders["train"],
            validation_loader=loaders["validation"],
            device=device,
            stage=stage,
            statistics=statistics,
        )

        # Добавляем результаты этапа в общую историю
        full_history.extend(stage_history)

        # Освобождаем память GPU перед следующим этапом
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return full_history


if __name__ == "__main__":
    # Запуск всех этапов напрямую из терминала:
    # python -m scripts.train
    train("config.yaml")