import os
import random
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


REQUIRED_DISH_COLUMNS = {
    "dish_id",
    "total_calories",
    "total_mass",
    "ingredients",
    "split",
}

REQUIRED_INGREDIENT_COLUMNS = {
    "id",
    "ingr",
}


def ingredient_id_to_code(ingredient_id: int) -> str:
    """Преобразует числовой ID в формат из dish.csv."""

    return f"ingr_{int(ingredient_id):010d}"


def load_and_prepare_data(
    data_dir: str | Path = "data",
    check_images: bool = True,
) -> pd.DataFrame:
    """
    Загружает CSV, заменяет ID ингредиентов названиями
    и добавляет пути к изображениям.
    """

    data_dir = Path(data_dir)

    dish_path = data_dir / "dish.csv"
    ingredients_path = data_dir / "ingredients.csv"
    images_dir = data_dir / "images"

    if not dish_path.is_file():
        raise FileNotFoundError(f"Не найден файл: {dish_path}")

    if not ingredients_path.is_file():
        raise FileNotFoundError(f"Не найден файл: {ingredients_path}")

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Не найден каталог: {images_dir}")

    dish_df = pd.read_csv(dish_path)
    ingredients_df = pd.read_csv(ingredients_path)

    missing_dish_columns = REQUIRED_DISH_COLUMNS - set(dish_df.columns)
    missing_ingredient_columns = REQUIRED_INGREDIENT_COLUMNS - set(
        ingredients_df.columns
    )

    if missing_dish_columns:
        raise ValueError(
            f"В dish.csv отсутствуют столбцы: {sorted(missing_dish_columns)}"
        )

    if missing_ingredient_columns:
        raise ValueError(
            "В ingredients.csv отсутствуют столбцы: "
            f"{sorted(missing_ingredient_columns)}"
        )

    if dish_df["dish_id"].duplicated().any():
        duplicate_count = int(dish_df["dish_id"].duplicated().sum())
        raise ValueError(f"Найдены дубликаты dish_id: {duplicate_count}")

    allowed_splits = {"train", "test"}
    actual_splits = set(dish_df["split"].dropna().unique())
    unknown_splits = actual_splits - allowed_splits

    if unknown_splits:
        raise ValueError(
            f"Найдены неизвестные значения split: {sorted(unknown_splits)}"
        )

    if (dish_df["total_mass"] <= 0).any():
        raise ValueError("В датасете обнаружена неположительная масса")

    if (dish_df["total_calories"] < 0).any():
        raise ValueError("В датасете обнаружена отрицательная калорийность")

    ingredient_mapping = {
        ingredient_id_to_code(row.id): str(row.ingr)
        for row in ingredients_df.itertuples(index=False)
    }

    def decode_ingredients(value: str) -> str:
        ingredient_ids = [
            item.strip()
            for item in str(value).split(";")
            if item.strip()
        ]

        unknown_ids = [
            ingredient_id
            for ingredient_id in ingredient_ids
            if ingredient_id not in ingredient_mapping
        ]

        if unknown_ids:
            raise ValueError(
                "В dish.csv найдены неизвестные ингредиенты: "
                f"{unknown_ids[:5]}"
            )

        ingredient_names = [
            ingredient_mapping[ingredient_id]
            for ingredient_id in ingredient_ids
        ]

        return ", ".join(ingredient_names)

    dish_df = dish_df.copy()

    dish_df["ingredient_text"] = dish_df["ingredients"].apply(
        decode_ingredients
    )

    dish_df["ingredient_count"] = dish_df["ingredients"].apply(
        lambda value: len(
            [
                item
                for item in str(value).split(";")
                if item.strip()
            ]
        )
    )

    dish_df["image_path"] = dish_df["dish_id"].apply(
        lambda dish_id: str(
            images_dir / str(dish_id) / "rgb.png"
        )
    )

    if check_images:
        image_exists = dish_df["image_path"].apply(
            lambda path: Path(path).is_file()
        )

        if not image_exists.all():
            missing_images = dish_df.loc[
                ~image_exists,
                ["dish_id", "image_path"],
            ]

            example_paths = missing_images["image_path"].head(5).tolist()

            raise FileNotFoundError(
                f"Не найдено изображений: {len(missing_images)}. "
                f"Примеры: {example_paths}"
            )

    return dish_df


def split_dataframe(
    dataframe: pd.DataFrame,
    validation_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделяет исходный train на train и validation.

    Официальный test остаётся отдельным и не участвует
    в выборе модели.
    """

    train_full = dataframe[
        dataframe["split"] == "train"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    # Для регрессии создаём интервалы по калорийности.
    # Это помогает сохранить похожее распределение target
    # в train и validation.
    calorie_bins = pd.qcut(
        train_full["total_calories"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    train_df, validation_df = train_test_split(
        train_full,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=calorie_bins,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, validation_df, test_df


def get_train_transforms(
    image_size: int = 224,
) -> A.Compose:
    """Аугментации для обучающей выборки."""

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def get_validation_transforms(
    image_size: int = 224,
) -> A.Compose:
    """Преобразования для validation и test."""

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


class DishDataset(Dataset):
    """Датасет фотографий, ингредиентов, массы и калорийности."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: A.Compose,
        mass_mean: float,
        mass_std: float,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.mass_mean = float(mass_mean)
        self.mass_std = float(mass_std)

        if self.mass_std <= 0:
            raise ValueError("Стандартное отклонение массы должно быть больше 0")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_array = np.asarray(image)

        transformed = self.transform(image=image_array)
        image_tensor = transformed["image"]

        raw_mass = float(row["total_mass"])
        normalized_mass = (
            raw_mass - self.mass_mean
        ) / self.mass_std

        target = float(row["total_calories"])

        return {
            "image": image_tensor,
            "text": row["ingredient_text"],
            "mass": torch.tensor(
                [normalized_mass],
                dtype=torch.float32,
            ),
            "raw_mass": torch.tensor(
                raw_mass,
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                target,
                dtype=torch.float32,
            ),
            "dish_id": row["dish_id"],
            "image_path": image_path,
        }


def seed_worker(worker_id: int) -> None:
    """Устанавливает seed для процесса DataLoader."""

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    dataframe: pd.DataFrame,
    batch_size: int = 4,
    image_size: int = 224,
    validation_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[
    dict[str, DataLoader],
    dict[str, pd.DataFrame],
    dict[str, float],
]:
 

    train_df, validation_df, test_df = split_dataframe(
        dataframe=dataframe,
        validation_size=validation_size,
        seed=seed,
    )

    # Статистики вычисляются только на train.
    mass_mean = float(train_df["total_mass"].mean())
    mass_std = float(train_df["total_mass"].std())

    train_dataset = DishDataset(
        dataframe=train_df,
        transform=get_train_transforms(image_size),
        mass_mean=mass_mean,
        mass_std=mass_std,
    )

    validation_dataset = DishDataset(
        dataframe=validation_df,
        transform=get_validation_transforms(image_size),
        mass_mean=mass_mean,
        mass_std=mass_std,
    )

    test_dataset = DishDataset(
        dataframe=test_df,
        transform=get_validation_transforms(image_size),
        mass_mean=mass_mean,
        mass_std=mass_std,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    common_parameters = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_parameters,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_parameters,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_parameters,
    )

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }

    dataframes = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    statistics = {
        "mass_mean": mass_mean,
        "mass_std": mass_std,
    }

    return loaders, dataframes, statistics