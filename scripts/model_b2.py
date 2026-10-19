import timm
from torch import nn
from transformers import AutoModel, AutoTokenizer

from scripts.model import CalorieModel


class CalorieModelB2(CalorieModel):
    """Отдельная версия модели с EfficientNet-B2."""

    def __init__(
        self,
        fine_tune_image=False,
    ) -> None:
        # Инициализируем nn.Module без создания EfficientNet-B0
        nn.Module.__init__(self)

        self.fine_tune_image = fine_tune_image

        # Более мощный визуальный энкодер
        self.image_encoder = timm.create_model(
            "efficientnet_b2",
            pretrained=True,
            num_classes=0,
        )

        image_features = self.image_encoder.num_features

        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 128),
            nn.ReLU(),
        )

        # Текстовый энкодер оставляем прежним
        text_model_name = "distilbert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_name
        )

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        text_features = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_features, 128),
            nn.ReLU(),
        )

        # Обработка массы
        self.mass_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        # Регрессионная голова
        self.regression_head = nn.Sequential(
            nn.Linear(128 + 128 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Замораживаем предобученные энкодеры
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        # При fine-tuning размораживаем последний блок B2
        if self.fine_tune_image:
            for parameter in self.image_encoder.blocks[-1].parameters():
                parameter.requires_grad = True