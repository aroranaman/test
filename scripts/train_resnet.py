#!/usr/bin/env python3
# Manthan/scripts/train_resnet.py
from __future__ import annotations
import os
import logging
from pathlib import Path
import glob
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import tensorflow as tf

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
NUM_CLASSES = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATCH_SIZE = 256

# --- Paths ---
DATA_DIR = Path("data/western_himalaya_tfrecords") # Update this to your pan-India TFRecord folder when ready
MODEL_PATH = Path("models/artifacts/resnet_western_himalaya_v1.pth")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Dataloader for Local TFRecord Files ---
def parse_tfrecord_fn(example):
    """Parses a single TFRecord example from our GEE export."""
    feature_description = {
        'array': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Assuming band order from GEE was B4 (Red), B3 (Green), B2 (Blue)
    image = tf.io.decode_raw(example['array'], tf.float32)
    image = tf.reshape(image, [3, PATCH_SIZE, PATCH_SIZE])
    
    label = tf.cast(example['label'], tf.int64)
    return image, label

def get_dataloader(file_list: list, is_train: bool):
    """Creates a PyTorch DataLoader from a list of TFRecord files."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if is_train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_tf = tf.data.TFRecordDataset(file_list, num_parallel_reads=tf.data.AUTOTUNE)
    if is_train:
        dataset_tf = dataset_tf.shuffle(1000)
    
    dataset_tf = dataset_tf.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return DataLoader(dataset_tf, batch_size=None, num_workers=0)

# --- Model Definition & Training Loop ---
def get_site_quality_model(num_classes: int = NUM_CLASSES):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_quality_model(train_loader, val_loader, num_epochs: int = NUM_EPOCHS):
    logging.info("Starting ResNet model training...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = get_site_quality_model(num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_iterator.set_postfix(loss=loss.item())

        scheduler.step()

    logging.info("Training complete.")
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    all_shards = sorted(glob.glob(str(DATA_DIR / "*.tfrecord")))
    if not all_shards:
        logging.error(f"No .tfrecord files found in {DATA_DIR}.")
        sys.exit(1)
        
    split_idx = int(len(all_shards) * 0.8)
    train_shards, val_shards = all_shards[:split_idx], all_shards[split_idx:]
    
    if not val_shards and train_shards:
        val_shards = train_shards 

    logging.info(f"Found {len(all_shards)} TFRecord shards. Using {len(train_shards)} for training and {len(val_shards)} for validation.")

    train_loader = get_dataloader(train_shards, is_train=True)
    val_loader = get_dataloader(val_shards, is_train=False)

    train_quality_model(train_loader, val_loader)