import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from scripts.dataset import (
    create_dataloaders,
    load_and_prepare_data,
)
from scripts.model import CalorieModel


def set_seed(seed):
    """Фиксирует генераторы случайных чисел."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_ridge_features(
    dataframe,
    ingredient_encoder,
    mass_mean,
    mass_std,
    fit_encoder=False,
    use_ingredient_mass_features=False,
):
    """
    Создаёт признаки для Ridge.

    Ингредиенты преобразуются в набор бинарных признаков,
    а масса стандартизируется статистиками train.
    """

    ingredient_lists = (
        dataframe["ingredient_text"]
        .str.split(", ")
    )

    if fit_encoder:
        ingredient_features = (
            ingredient_encoder.fit_transform(
                ingredient_lists
            )
        )
    else:
        ingredient_features = (
            ingredient_encoder.transform(
                ingredient_lists
            )
        )

    normalized_mass = (
        (
            dataframe["total_mass"].to_numpy()
            - mass_mean
        )
        / mass_std
    ).reshape(-1, 1)

    feature_blocks = [
        ingredient_features,
        csr_matrix(normalized_mass),
    ]

    if use_ingredient_mass_features:
        # Для каждого ингредиента добавляем взаимодействие
        # с массой порции. Это позволяет Ridge учитывать,
        # что одинаковая масса разных продуктов имеет
        # различную калорийность.
        ingredient_mass_features = (
            ingredient_features.multiply(normalized_mass)
        )
        feature_blocks.append(ingredient_mass_features)

    features = hstack(feature_blocks).tocsr()

    return features


def prepare_ridge_predictions(
    dataframes,
    statistics,
    ridge_alpha,
    ridge_folds,
    seed,
    ridge_model_path,
    use_ingredient_mass_features=False,
):
    """
    Обучает Ridge и подготавливает базовые прогнозы.

    Для train используются OOF-прогнозы, чтобы Ridge
    не предсказывал объекты, на которых сам обучался.

    Для validation используется Ridge, обученный
    на всей обучающей выборке.
    """

    train_df = dataframes["train"]
    validation_df = dataframes["validation"]

    mass_mean = statistics["mass_mean"]
    mass_std = statistics["mass_std"]

    ingredient_encoder = MultiLabelBinarizer(
        sparse_output=True
    )

    x_train = create_ridge_features(
        dataframe=train_df,
        ingredient_encoder=ingredient_encoder,
        mass_mean=mass_mean,
        mass_std=mass_std,
        fit_encoder=True,
        use_ingredient_mass_features=(
            use_ingredient_mass_features
        ),
    )

    x_validation = create_ridge_features(
        dataframe=validation_df,
        ingredient_encoder=ingredient_encoder,
        mass_mean=mass_mean,
        mass_std=mass_std,
        use_ingredient_mass_features=(
            use_ingredient_mass_features
        ),
    )

    if use_ingredient_mass_features:
        print(
            "Включены признаки: "
            "ингредиент × масса"
        )

    print("Ridge признаков:", x_train.shape[1])

    y_train = train_df[
        "total_calories"
    ].to_numpy(dtype=np.float32)

    y_validation = validation_df[
        "total_calories"
    ].to_numpy(dtype=np.float32)

    # Здесь будут храниться OOF-прогнозы для train
    train_predictions = np.zeros(
        len(train_df),
        dtype=np.float32,
    )

    kfold = KFold(
        n_splits=ridge_folds,
        shuffle=True,
        random_state=seed,
    )

    # Каждый объект train предсказывается моделью,
    # которая не обучалась на этом объекте
    for train_indices, holdout_indices in kfold.split(
        x_train
    ):
        fold_model = Ridge(
            alpha=ridge_alpha
        )

        fold_model.fit(
            x_train[train_indices],
            y_train[train_indices],
        )

        train_predictions[holdout_indices] = (
            fold_model.predict(
                x_train[holdout_indices]
            )
        )

    # Итоговый Ridge обучается на всём train
    ridge_model = Ridge(
        alpha=ridge_alpha
    )

    ridge_model.fit(
        x_train,
        y_train,
    )

    validation_predictions = ridge_model.predict(
        x_validation
    ).astype(np.float32)

    train_mae = np.mean(
        np.abs(train_predictions - y_train)
    )

    validation_mae = np.mean(
        np.abs(
            validation_predictions
            - y_validation
        )
    )

    print(
        f"Ridge OOF train MAE: {train_mae:.2f}"
    )

    print(
        "Ridge validation MAE: "
        f"{validation_mae:.2f}"
    )

    # Связываем прогноз Ridge с dish_id
    prediction_maps = {
        "train": {
            str(dish_id): float(prediction)
            for dish_id, prediction in zip(
                train_df["dish_id"],
                train_predictions,
            )
        },
        "validation": {
            str(dish_id): float(prediction)
            for dish_id, prediction in zip(
                validation_df["dish_id"],
                validation_predictions,
            )
        },
    }

    # Сохраняем Ridge и преобразователь ингредиентов.
    # Они понадобятся для итогового тестирования.
    ridge_model_path = Path(ridge_model_path)

    ridge_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "ridge_model": ridge_model,
            "ingredient_encoder": ingredient_encoder,
            "mass_mean": mass_mean,
            "mass_std": mass_std,
            "ridge_alpha": ridge_alpha,
            "use_ingredient_mass_features": (
                use_ingredient_mass_features
            ),
        },
        ridge_model_path,
    )

    print(
        "Модель Ridge сохранена:",
        ridge_model_path,
    )

    return prediction_maps


def get_ridge_batch_predictions(
    dish_ids,
    prediction_map,
    device,
):
    """Возвращает прогнозы Ridge для текущего batch."""

    predictions = [
        prediction_map[str(dish_id)]
        for dish_id in dish_ids
    ]

    return torch.tensor(
        predictions,
        dtype=torch.float32,
        device=device,
    )


def train_residual_model(
    model,
    train_loader,
    validation_loader,
    ridge_predictions,
    device,
    config,
    statistics,
):
    """
    Обучает нейросеть предсказывать поправку к Ridge.

    Итоговый прогноз равен сумме прогноза Ridge
    и поправки мультимодальной нейросети.
    """

    criterion = nn.L1Loss()

    # В optimizer передаются все обучаемые параметры
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=config["learning_rate"],
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_path = Path(
        config["checkpoint_path"]
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    for epoch in range(config["epochs"]):
        model.train()

        # BatchNorm предобученного EfficientNet
        # не изменяет накопленные статистики
        model.image_encoder.eval()
        model.text_encoder.eval()

        train_error = 0.0

        for batch in tqdm(
            train_loader,
            desc=(
                "residual_learning: "
                f"эпоха {epoch + 1}/"
                f"{config['epochs']}"
            ),
        ):
            images = batch["image"].to(device)
            texts = batch["text"]
            masses = batch["mass"].to(device)
            targets = batch["target"].to(device)

            ridge_batch = get_ridge_batch_predictions(
                dish_ids=batch["dish_id"],
                prediction_map=(
                    ridge_predictions["train"]
                ),
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                # Нейросеть предсказывает поправку
                corrections = model(
                    images=images,
                    texts=texts,
                    masses=masses,
                )

                # Итоговый прогноз объединяет
                # Ridge и нейросетевую поправку
                predictions = (
                    ridge_batch + corrections
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

                ridge_batch = (
                    get_ridge_batch_predictions(
                        dish_ids=batch["dish_id"],
                        prediction_map=(
                            ridge_predictions[
                                "validation"
                            ]
                        ),
                        device=device,
                    )
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    corrections = model(
                        images=images,
                        texts=texts,
                        masses=masses,
                    )

                    predictions = (
                        ridge_batch + corrections
                    )

                validation_error += (
                    torch.abs(
                        predictions - targets
                    )
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
                "validation_mae": (
                    validation_mae
                ),
            }
        )

        print(
            f"Эпоха {epoch + 1}: "
            f"train MAE = {train_mae:.2f}, "
            "validation MAE = "
            f"{validation_mae:.2f}"
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            epochs_without_improvement = 0

            torch.save(
                {
                    "stage": "residual_learning",
                    "epoch": epoch + 1,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "validation_mae": (
                        validation_mae
                    ),
                    "mass_mean": (
                        statistics["mass_mean"]
                    ),
                    "mass_std": (
                        statistics["mass_std"]
                    ),
                    "ridge_model_path": (
                        config["ridge_model_path"]
                    ),
                },
                checkpoint_path,
            )

            print(
                "Сохранена лучшая модель: "
                f"MAE = {validation_mae:.2f}"
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config["patience"]
        ):
            print(
                "Обучение остановлено "
                "по early stopping"
            )
            break

    return history


def train_residual(config_path):
    """
    Запускает отдельное обучение поправки к Ridge.

    Обычный scripts/train.py при этом
    не используется и не изменяется.
    """

    with open(
        config_path,
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    set_seed(config["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataframe = load_and_prepare_data(
        data_dir=config["data_dir"],
    )

    loaders, dataframes, statistics = (
        create_dataloaders(
            dataframe=dataframe,
            batch_size=config["batch_size"],
            image_size=config["image_size"],
            validation_size=(
                config["validation_size"]
            ),
            seed=config["seed"],
            num_workers=config["num_workers"],
        )
    )

    print("Устройство:", device)
    print("Train:", len(dataframes["train"]))
    print(
        "Validation:",
        len(dataframes["validation"]),
    )

    # Создаём честные базовые прогнозы Ridge
    ridge_predictions = (
        prepare_ridge_predictions(
            dataframes=dataframes,
            statistics=statistics,
            ridge_alpha=config["ridge_alpha"],
            ridge_folds=config["ridge_folds"],
            seed=config["seed"],
            ridge_model_path=(
                config["ridge_model_path"]
            ),
            use_ingredient_mass_features=config.get(
                "use_ingredient_mass_features",
                False,
            ),
        )
    )

    # Создаём мультимодальную модель
    model = CalorieModel(
        fine_tune_image=(
            config["fine_tune_image"]
        ),
    ).to(device)

    initial_checkpoint = Path(
        config["initial_checkpoint"]
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

    # Последний слой обнуляется.
    # До начала обучения поправка равна нулю,
    # поэтому итоговый прогноз равен Ridge.
    output_layer = model.regression_head[-1]

    nn.init.zeros_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)

    history = train_residual_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        ridge_predictions=ridge_predictions,
        device=device,
        config=config,
        statistics=statistics,
    )

    return history


if __name__ == "__main__":
    # Запуск отдельного обучения поправки:
    # python -m scripts.train_residual
    train_residual("config_residual.yaml")
