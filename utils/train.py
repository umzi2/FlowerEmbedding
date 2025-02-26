 
import numpy as np
import torch
import torch.nn as nn
from pepeline import noise_generate, TypeNoise
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from utils.adan_sf import adan_sf
from utils.dataloader import ImageDataset
import torch.multiprocessing as mp
from utils.mamba_out import MambaOut
import torch.nn.functional
import random


class CutOut(nn.Module):
    """
    Применяет операцию CutOut, которая случайным образом заменяет части изображения шумом или фиксированным цветом.
    """

    def __init__(self, probability=0.5):
        """
        :param probability: Вероятность применения CutOut.
        """
        super().__init__()
        self.p = probability

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Применяет операцию CutOut к входному тензору.

        :param tensor: Изображение в виде тензора с формой (C, H, W).
        :return: Изображение с применённым эффектом CutOut.
        """
        c, h, w = tensor.shape

        # Если случайное число больше или равно вероятности, возвращаем исходный тензор
        if torch.rand(1) >= self.p:
            return tensor

        # Генерируем маску с использованием функции noise_generate (тип шума SIMPLEX)
        mask = noise_generate((c, h, w), TypeNoise.SIMPLEX, 1, 0.02, 0.1) > 0.3
        tensor = tensor.clone()
        # С вероятностью 50% заменяем пиксели в области маски случайными значениями
        if torch.rand(1) >= 0.5:
            noise = torch.rand((c, h, w), device=tensor.device, dtype=tensor.dtype)
            tensor[mask] = noise[mask]
        else:
            # Иначе заменяем пиксели в области маски на случайный цвет для каждого канала
            random_color = torch.rand(c, 1, 1, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.where(
                torch.tensor(mask, device=tensor.device, dtype=torch.bool),
                random_color,
                tensor,
            )
        return tensor


class Random90DegreeRotation:
    @staticmethod
    def __call__(x):
        num_rotations = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, num_rotations, [1, 2])


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.01):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.lambda_c = lambda_c

    def forward(self, features, labels):
        batch_centers = self.centers[labels]  # Берём центры классов
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()  # L2
        return self.lambda_c * loss


class ImageClassifierTrainer:
    def __init__(
        self,
        model,
        # Основные параметры
        train_data_path: str,
        val_data_path: str,
        checkpoint_dir: str,
        num_classes: int,
        # Параметры обучения
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        num_epochs: int = 1000,
        # Параметры аугментаций
        flip_prob: float = 0.5,
        use_cutout: bool = True,
        # Параметры модели и loss
        embedding_size: int = 128,
        center_loss_weight: float = 1.0,
        # Параметры оптимизатора
        optimizer_betas: list = [0.98, 0.92, 0.99],
        optimizer_weight_decay: float = 0.01,
        optimizer_schedule_free: bool = True,
        # Параметры DataLoader
        num_workers: int = 16,
        persistent_workers: bool = True,
        # Настройки воспроизводимости
        seed: int = None,
        # Настройки градиентов
        grad_clip_max_norm: float = 1.0,
        # Настройки смешанной точности
        use_amp: bool = True,
    ):
        # Инициализация воспроизводимости
        if seed is not None:
            self._set_seed(seed)

        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mp.set_start_method("spawn", force=True)

        # Инициализация модели
        self.model = model.to(self.device)

        # Потери
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion2 = CenterLoss(num_classes, embedding_size).cuda()
        self.center_loss_weight = center_loss_weight

        # Оптимизатор
        self.optimizer = adan_sf(
            self.model.parameters(),
            learning_rate,
            betas=optimizer_betas,
            weight_decay=optimizer_weight_decay,
            schedule_free=optimizer_schedule_free,
        )

        # Настройки обучения
        self.scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.grad_clip_max_norm = grad_clip_max_norm

        # Аугментации
        train_transform = self._get_train_transforms(flip_prob, use_cutout)

        # DataLoaders
        self.train_loader = DataLoader(
            ImageDataset(train_data_path, True, train_transform),
            batch_size=batch_size,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_init_fn if seed else None,
        )

        self.val_loader = DataLoader(
            ImageDataset(val_data_path, val=True)
        )

        # Пути
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _set_seed(self, seed: int):
        """Устанавливает сид для всех генераторов случайных чисел"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _worker_init_fn(self, worker_id):
        """Инициализация воркеров DataLoader с фиксированным сидом"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _get_train_transforms(self, flip_prob: float, use_cutout: bool):
        """Создает трансформации с настраиваемыми параметрами"""
        transform_list = [
            transforms.RandomVerticalFlip(flip_prob),
            transforms.RandomHorizontalFlip(flip_prob),
            Random90DegreeRotation(),
            transforms.ColorJitter(0.25,0.25,0.25,0.25),
            transforms.RandomGrayscale(),
        ]

        if use_cutout:
            transform_list.append(CutOut())

        return transforms.Compose(transform_list)

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels, _ in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.scaler_g.is_enabled()):
                    outputs, embed = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss += self.center_loss_weight * self.criterion2(embed, labels)

                self.scaler_g.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_max_norm
                )
                self.scaler_g.step(self.optimizer)
                self.scaler_g.update()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}"
            )

            # Валидация
            val_loss = self.validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, best=True)
                print("Best checkpoint saved!")

            self.save_checkpoint(epoch, val_loss, best=False)
            print("Regular checkpoint saved!")
            torch.cuda.empty_cache()

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, image_name in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # print(images.shape,labels.shape)
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Преобразуем выход модели в предсказанные классы
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        del images, labels
        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total  # Вычисляем точность

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_val_loss

    def save_checkpoint(self, epoch, loss, best=True):
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"epoch_{epoch}.pth",
        )

        if best:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                "best_model.pth",
            )
        if best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )
        else:
            torch.save({"model_state_dict": self.model.state_dict()}, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}"
        )



