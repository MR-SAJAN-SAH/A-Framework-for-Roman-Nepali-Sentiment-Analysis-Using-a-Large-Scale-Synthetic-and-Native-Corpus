import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    model_name = "xlm-roberta-base"
    hidden_size = 768
    num_sentiment_classes = 3
    num_script_classes = 3
    contrastive_proj_size = 128
    translit_hidden_size = 256
    max_length = 128

    # Training
    batch_size = 32  # Reduced for stability
    accumulation_steps = 2
    num_epochs = 6
    learning_rate = 2e-5
    head_learning_rate = 5e-5
    warmup_ratio = 0.08
    max_grad_norm = 1.0

    # Loss weights
    lambda_sentiment = 1.0
    lambda_transliteration = 0.5
    lambda_contrastive = 0.5
    lambda_script = 0.2
    lambda_adversarial = 0.1

    # Contrastive
    temperature = 0.07

    # Data
    sentiment_labels = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    script_labels = {'roman': 0, 'devanagari': 1, 'english': 2}

    # Paths - Use relative paths for Python files
    model_save_path = "./best_model"
    checkpoint_path = "./checkpoints"

# Custom Dataset
class RomanNepaliDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config, augment=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.config = config
        self.augment = augment
        self.texts = dataframe['text'].tolist()
        self.sentiments = dataframe['sentiment'].map(config.sentiment_labels).tolist()

    def __len__(self):
        return len(self.data)

    def augment_text(self, text):
        """Simple text augmentation for Roman Nepali"""
        variations = [
            text,
            text.replace('aa', 'a').replace('ee', 'e').replace('oo', 'o'),
            text.replace('ch', 'c').replace('sh', 's'),
            text.replace('4', 'for').replace('2', 'to'),
            text.replace('u', 'oo').replace('i', 'ee'),
        ]
        return np.random.choice(variations)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.augment and np.random.random() < 0.3:
            text = self.augment_text(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        sentiment = self.sentiments[idx]
        script_id = self.config.script_labels['roman']

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_labels': torch.tensor(sentiment, dtype=torch.long),
            'script_labels': torch.tensor(script_id, dtype=torch.long),
            'text': text
        }

# Model Components
class ContrastiveProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.contrastive_proj_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class DomainDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_script_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class TransliterationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.translit_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        self.output_proj = nn.Linear(config.translit_hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        expanded = pooled.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        decoder_out, _ = self.decoder(expanded)
        output = self.output_proj(decoder_out)
        return output

# Main Model
class MultiTaskNepaliSentimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = XLMRobertaModel.from_pretrained(config.model_name)
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.num_sentiment_classes)
        self.script_classifier = nn.Linear(config.hidden_size, config.num_script_classes)
        self.contrastive_head = ContrastiveProjectionHead(config)
        self.transliteration_head = TransliterationHead(config)
        self.domain_discriminator = DomainDiscriminator(config)
        self.dropout = nn.Dropout(0.1)
        self._init_weights([self.sentiment_classifier, self.script_classifier])

    def _init_weights(self, modules):
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        sequence_output = outputs.last_hidden_state

        return {
            'pooled_output': pooled_output,
            'sequence_output': sequence_output,
            'hidden_states': outputs.hidden_states
        }

    def get_sentiment_logits(self, pooled_output):
        return self.sentiment_classifier(pooled_output)

    def get_script_logits(self, pooled_output):
        return self.script_classifier(pooled_output)

    def get_contrastive_embedding(self, pooled_output):
        return self.contrastive_head(pooled_output)

    def get_transliteration_output(self, sequence_output):
        return self.transliteration_head(sequence_output)

    def get_domain_logits(self, pooled_output):
        return self.domain_discriminator(pooled_output)

# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# Contrastive Loss
def compute_contrastive_loss(embeddings, labels, temperature=0.07):
    batch_size = embeddings.size(0)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()
    mask = mask - torch.eye(batch_size, device=mask.device)
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

# Training Utilities
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.sentiment_losses = []
        self.contrastive_losses = []
        self.script_losses = []
        self.adversarial_losses = []
        self.transliteration_losses = []
        self.accuracies = []
        self.f1_scores = []

    def update(self, metrics):
        self.losses.append(metrics.get('loss', 0))
        self.sentiment_losses.append(metrics.get('sentiment_loss', 0))
        self.contrastive_losses.append(metrics.get('contrastive_loss', 0))
        self.script_losses.append(metrics.get('script_loss', 0))
        self.adversarial_losses.append(metrics.get('adversarial_loss', 0))
        self.transliteration_losses.append(metrics.get('transliteration_loss', 0))
        self.accuracies.append(metrics.get('accuracy', 0))
        self.f1_scores.append(metrics.get('f1', 0))

# Evaluation Function without tqdm
def evaluate_model(model, data_loader, device, config):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    criterion_ce = nn.CrossEntropyLoss()

    print(f"Evaluating {len(data_loader)} batches...")
    eval_start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)

            outputs = model(input_ids, attention_mask)
            pooled_output = outputs['pooled_output']
            sentiment_logits = model.get_sentiment_logits(pooled_output)
            loss = criterion_ce(sentiment_logits, sentiment_labels)

            total_loss += loss.item()
            predictions = torch.argmax(sentiment_logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(sentiment_labels.cpu().numpy())

            # Print progress every 20 steps
            if (step + 1) % 20 == 0 or (step + 1) == len(data_loader):
                progress = (step + 1) / len(data_loader) * 100
                print(f"Evaluation progress: {progress:.1f}% ({step+1}/{len(data_loader)})")

    eval_time = time.time() - eval_start_time
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Evaluation completed in {eval_time:.2f}s")
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

# Training Function without tqdm
def train_phase1(model, train_loader, val_loader, config, device):
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Optimizers
    encoder_params = list(model.encoder.named_parameters())
    head_params = []
    for name, param in model.named_parameters():
        if not name.startswith('encoder'):
            head_params.append(param)

    optimizer = AdamW([
        {'params': [p for n, p in encoder_params], 'lr': config.learning_rate},
        {'params': head_params, 'lr': config.head_learning_rate}
    ])

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    gradient_reversal = GradientReversal(alpha=config.lambda_adversarial)

    # Training tracking
    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(patience=3, path=os.path.join(config.model_save_path, 'best_model.pt'))

    global_step = 0
    best_f1 = 0

    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*60}")

        # Training phase
        model.train()
        total_loss = 0
        train_start_time = time.time()
        batch_times = []

        print(f"Training batches: {len(train_loader)}")
        print("-" * 80)

        for step, batch in enumerate(train_loader):
            batch_start_time = time.time()

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            script_labels = batch['script_labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            pooled_output = outputs['pooled_output']
            sequence_output = outputs['sequence_output']

            # Calculate losses
            sentiment_logits = model.get_sentiment_logits(pooled_output)
            sentiment_loss = criterion_ce(sentiment_logits, sentiment_labels)

            script_logits = model.get_script_logits(pooled_output)
            script_loss = criterion_ce(script_logits, script_labels)

            contrastive_embeddings = model.get_contrastive_embedding(pooled_output)
            contrastive_loss_val = compute_contrastive_loss(contrastive_embeddings, sentiment_labels, config.temperature)

            reversed_features = gradient_reversal(pooled_output)
            domain_logits = model.get_domain_logits(reversed_features)
            adversarial_loss = criterion_ce(domain_logits, script_labels)

            transliteration_output = model.get_transliteration_output(sequence_output)
            transliteration_loss = criterion_mse(transliteration_output, sequence_output.detach())

            # Combined loss
            loss = (config.lambda_sentiment * sentiment_loss +
                   config.lambda_script * script_loss +
                   config.lambda_contrastive * contrastive_loss_val -
                   config.lambda_adversarial * adversarial_loss +
                   config.lambda_transliteration * transliteration_loss)

            # Backward pass
            loss = loss / config.accumulation_steps
            loss.backward()

            if (step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.accumulation_steps
            global_step += 1

            # Calculate time metrics
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            # Calculate ETA
            if len(batch_times) > 10:
                avg_batch_time = np.mean(batch_times[-10:])
            else:
                avg_batch_time = np.mean(batch_times)

            remaining_batches = len(train_loader) - step - 1
            time_remaining = avg_batch_time * remaining_batches
            mins_remaining = int(time_remaining // 60)
            secs_remaining = int(time_remaining % 60)

            # Print progress every 20 steps
            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                progress_percent = (step + 1) / len(train_loader) * 100
                current_lr = scheduler.get_last_lr()[0]

                print(f"Step {step+1:4d}/{len(train_loader)} [{progress_percent:5.1f}%] | "
                      f"Loss: {loss.item() * config.accumulation_steps:7.4f} | "
                      f"Sent: {sentiment_loss.item():6.4f} | "
                      f"Cont: {contrastive_loss_val.item():6.4f} | "
                      f"LR: {current_lr:.1e} | "
                      f"ETA: {mins_remaining:02d}:{secs_remaining:02d}")

        avg_train_loss = total_loss / len(train_loader)
        train_time = time.time() - train_start_time
        train_mins = int(train_time // 60)
        train_secs = int(train_time % 60)

        print("-" * 80)
        print(f"Training completed in {train_mins:02d}:{train_secs:02d}")

        # Validation phase
        print("\nStarting validation...")
        val_metrics = evaluate_model(model, val_loader, device, config)
        val_metrics['train_loss'] = avg_train_loss
        val_metrics['epoch'] = epoch + 1
        val_metrics['train_time'] = train_time

        # Update metrics
        metrics_tracker.update(val_metrics)

        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss:    {avg_train_loss:.4f}")
        print(f"   Val Loss:      {val_metrics['loss']:.4f}")
        print(f"   Val Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"   Val F1-Score:  {val_metrics['f1']:.4f}")
        print(f"   Train Time:    {train_mins:02d}:{train_secs:02d}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config
            }, os.path.join(config.model_save_path, 'best_model.pt'))
            print(f"✅ New best model saved with F1: {best_f1:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'config': config,
            'metrics': val_metrics
        }
        torch.save(checkpoint, os.path.join(config.checkpoint_path, f'checkpoint_epoch_{epoch+1}.pt'))

        # Early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    return metrics_tracker

# Visualization Functions
def plot_training_metrics(metrics_tracker, config):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(metrics_tracker.losses, label='Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(metrics_tracker.sentiment_losses, label='Sentiment Loss', alpha=0.7)
    axes[0, 1].plot(metrics_tracker.contrastive_losses, label='Contrastive Loss', alpha=0.7)
    axes[0, 1].plot(metrics_tracker.script_losses, label='Script Loss', alpha=0.7)
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()

    axes[0, 2].plot(metrics_tracker.accuracies, label='Accuracy', color='green')
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].legend()

    axes[1, 0].plot(metrics_tracker.f1_scores, label='F1 Score', color='orange')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()

    axes[1, 1].plot(metrics_tracker.adversarial_losses, label='Adversarial Loss', color='red')
    axes[1, 1].set_title('Adversarial Loss')
    axes[1, 1].legend()

    axes[1, 2].plot(metrics_tracker.transliteration_losses, label='Transliteration Loss', color='purple')
    axes[1, 2].set_title('Transliteration Loss')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_path, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(labels, predictions, class_names, config):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(config.model_save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_model_info(model, config):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print("MODEL INFORMATION")
    print(f"{'='*50}")
    print(f"Base Model: {config.model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Max Sequence Length: {config.max_length}")
    print(f"{'='*50}")

# Main Execution
def main():
    # Force GPU usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Using CPU")

    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('RomanNepali.csv')
        print(f"Dataset size: {len(df)}")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    except FileNotFoundError:
        print("❌ Error: RomanNepali.csv not found!")
        print("Please make sure the file is in the same directory as this script.")
        return

    # Split data
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Initialize model and tokenizer
    config = Config()
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.model_name)

    # Create datasets and loaders
    train_dataset = RomanNepaliDataset(train_df, tokenizer, config, augment=True)
    val_dataset = RomanNepaliDataset(val_df, tokenizer, config, augment=False)
    test_dataset = RomanNepaliDataset(test_df, tokenizer, config, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("Initializing model...")
    model = MultiTaskNepaliSentimentModel(config)
    model.to(device)

    print_model_info(model, config)

    # Train model
    print("Starting Phase 1 training...")
    start_time = time.time()
    metrics_tracker = train_phase1(model, train_loader, val_loader, config, device)
    total_training_time = time.time() - start_time

    print(f"\nTotal training time: {total_training_time/60:.2f} minutes")

    # Final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(config.model_save_path, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Final evaluation on test set...")
    test_metrics = evaluate_model(model, test_loader, device, config)

    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")

    # Classification report
    class_names = list(config.sentiment_labels.keys())
    print(f"\nClassification Report:")
    print(classification_report(test_metrics['labels'], test_metrics['predictions'],
                              target_names=class_names, digits=4))

    # Save results
    results = {
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_loss': test_metrics['loss'],
        'best_epoch': checkpoint.get('epoch', 'unknown'),
        'training_time': total_training_time,
        'config': config.__dict__
    }

    with open(os.path.join(config.model_save_path, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved in: {config.model_save_path}")

    # Generate plots
    print("Generating visualizations...")
    plot_training_metrics(metrics_tracker, config)
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], class_names, config)

if __name__ == "__main__":
    main()