import timm
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CalorieModel(nn.Module):
    """Мультимодальная модель для предсказания калорийности."""

    def __init__(
        self,
        fine_tune_image=False,
    ) -> None:
        super().__init__()

        self.fine_tune_image = fine_tune_image

        # Энкодер изображения
        self.image_encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
        )


        image_features = self.image_encoder.num_features

        self.image_projection = nn.Sequential(
            nn.Linear(image_features, 128),
            nn.ReLU(),
        )

        # Энкодер текста
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

        # Объединение признаков
        self.regression_head = nn.Sequential(
            nn.Linear(128 + 128 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # По умолчанию предобученные энкодеры заморожены
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        # При дообучении размораживаем последние слои EfficientNet
        if self.fine_tune_image:
            for parameter in self.image_encoder.blocks[-1].parameters():
                parameter.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """Выполняет прямой проход модели."""

        # Для размороженных слоёв изображения нужны градиенты
        image_features = self.image_encoder(images)

        # Текстовый энкодер остаётся замороженным
        with torch.no_grad():
            tokens = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )

            tokens = {
                name: value.to(images.device)
                for name, value in tokens.items()
            }

            text_output = self.text_encoder(**tokens)
            text_features = text_output.last_hidden_state[:, 0]

        image_features = self.image_projection(
            image_features
        )

        text_features = self.text_projection(
            text_features
        )

        mass_features = self.mass_encoder(
            masses
        )

        combined_features = torch.cat(
            [
                image_features,
                text_features,
                mass_features,
            ],
            dim=1,
        )

        predictions = self.regression_head(
            combined_features
        )

        return predictions.squeeze(1)