import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from src.utils import AverageMeter, get_lr


def _vram_str():
    if not torch.cuda.is_available():
        return ""
    used  = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{used:.1f}/{total:.1f}GB"


def train_epoch_baseline(model, dataloader, criterion, optimizer, device,
                         epoch, config, scaler=None):
    model.train()
    use_amp = getattr(config, 'USE_AMP', False) and scaler is not None

    losses  = AverageMeter()
    correct = 0
    total   = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) 

        if use_amp:
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc':  f'{100.*correct/total:.2f}%',
            'lr':   f'{get_lr(optimizer):.6f}',
            'vram': _vram_str()
        })

    return losses.avg, 100. * correct / total


def train_epoch_residual_spatial(model, dataloader, criterion, optimizer, device,
                                 epoch, config, residual_extractor, scaler=None):
    model.train()
    use_amp = getattr(config, 'USE_AMP', False) and scaler is not None

    losses  = AverageMeter()
    correct = 0
    total   = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        residual_images = residual_extractor.extract_batch(images)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                outputs = model(residual_images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(residual_images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc':  f'{100.*correct/total:.2f}%',
            'lr':   f'{get_lr(optimizer):.6f}',
            'vram': _vram_str()
        })

    return losses.avg, 100. * correct / total


def train_epoch_residual_dct(model, dataloader, criterion, optimizer, device,
                             epoch, config, extractors, scaler=None):
    model.train()
    use_amp = getattr(config, 'USE_AMP', False) and scaler is not None

    residual_extractor = extractors['residual']
    dct_extractor      = extractors['dct']

    losses  = AverageMeter()
    correct = 0
    total   = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)

        residual_images = residual_extractor.extract_batch(images)
        dct_images      = dct_extractor.extract_batch(residual_images)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                outputs = model(dct_images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(dct_images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, 'GRADIENT_CLIP', 1.0)
            )
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc':  f'{100.*correct/total:.2f}%',
            'lr':   f'{get_lr(optimizer):.6f}',
            'vram': _vram_str()
        })

    return losses.avg, 100. * correct / total



def plot_epoch_progress(history, epoch, method, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'],   'r-s', label='Val Loss',   linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Progress — Epoch {epoch}', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'],   'r-s', label='Val Acc',   linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy Progress — Epoch {epoch}', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{method}_epoch_{epoch:02d}_progress.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                    history, best_acc, config, method):
    checkpoint_path = config.get_checkpoint_path(method, epoch)

    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict() if scaler else None,
        'history':              history,
        'best_acc':             best_acc,
    }
    torch.save(checkpoint, checkpoint_path)

    state_path = os.path.join(config.CHECKPOINTS_DIR, f'{method}_training_state.json')
    with open(state_path, 'w') as f:
        json.dump({
            'last_epoch':   epoch,
            'total_epochs': config.NUM_EPOCHS,
            'best_acc':     best_acc,
            'completed':    (epoch == config.NUM_EPOCHS)
        }, f, indent=4)

    return checkpoint_path


def load_checkpoint(config, method, model, optimizer, scheduler, scaler=None):
    state_path = os.path.join(config.CHECKPOINTS_DIR, f'{method}_training_state.json')

    empty = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if not os.path.exists(state_path):
        return None, 0, empty, 0.0

    with open(state_path) as f:
        state = json.load(f)

    last_epoch = state['last_epoch']

    if state.get('completed', False):
        return None, last_epoch, None, state['best_acc']

    checkpoint_path = config.get_checkpoint_path(method, last_epoch)
    if not os.path.exists(checkpoint_path):
        return None, 0, empty, 0.0

    print(f"\n{'='*70}")
    print(f"RESUMING TRAINING FROM EPOCH {last_epoch}".center(70))
    print(f"{'='*70}")

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    history  = checkpoint['history']
    best_acc = checkpoint['best_acc']

    print(f"Resumed from epoch {last_epoch}, best acc so far: {best_acc:.2f}%")
    print(f"{'='*70}\n")

    return checkpoint, last_epoch, history, best_acc

def train_model(model, train_loader, val_loader, test_loader,
                config, method, feature_extractor=None, resume=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    use_amp = getattr(config, 'USE_AMP', False) and torch.cuda.is_available()
    scaler  = GradScaler() if use_amp else None

    if use_amp:
        print(f"\n[GPU] Mixed Precision (AMP) AKTIF — training lebih cepat!")
    else:
        print(f"\n[GPU] AMP tidak aktif (CPU mode atau USE_AMP=False)")

    start_epoch = 1
    history     = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc    = 0.0

    if resume:
        checkpoint, start_epoch, loaded_history, loaded_best_acc = load_checkpoint(
            config, method, model, optimizer, scheduler, scaler
        )
        if checkpoint is not None:
            history     = loaded_history
            best_acc    = loaded_best_acc
            start_epoch += 1
        elif loaded_history is None:
            print(f"\n{method} training already completed!")
            return history

    epoch_progress_dir = os.path.join(config.FIGURES_DIR, f'{method}_epoch_progress')
    os.makedirs(epoch_progress_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"TRAINING — {method.upper()}".center(70))
    print("="*70)
    print(f"Epochs   : {start_epoch} → {config.NUM_EPOCHS}")
    print(f"Device   : {config.DEVICE}")
    print(f"AMP      : {'ON' if use_amp else 'OFF'}")
    print(f"Batch    : {config.BATCH_SIZE}")
    if torch.cuda.is_available():
        print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("="*70)

    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if method == 'baseline':
                train_loss, train_acc = train_epoch_baseline(
                    model, train_loader, criterion, optimizer,
                    config.DEVICE, epoch, config, scaler
                )
            elif method == 'residual_spatial':
                train_loss, train_acc = train_epoch_residual_spatial(
                    model, train_loader, criterion, optimizer,
                    config.DEVICE, epoch, config, feature_extractor, scaler
                )
            elif method == 'residual_dct':
                train_loss, train_acc = train_epoch_residual_dct(
                    model, train_loader, criterion, optimizer,
                    config.DEVICE, epoch, config, feature_extractor, scaler
                )

            from src.evaluate import evaluate_model
            val_results = evaluate_model(
                model, val_loader, config, method, feature_extractor, verbose=False
            )
            val_loss = val_results['loss']
            val_acc  = val_results['accuracy']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            scheduler.step(val_acc)

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}/{config.NUM_EPOCHS} SUMMARY".center(70))
            print(f"{'='*70}")
            print(f"{'Train Loss:':<20} {train_loss:>10.4f}  |  "
                  f"{'Train Acc:':<15} {train_acc:>8.2f}%")
            print(f"{'Val Loss:':<20} {val_loss:>10.4f}  |  "
                  f"{'Val Acc:':<15} {val_acc:>8.2f}%")
            print(f"{'Learning Rate:':<20} {get_lr(optimizer):>10.6f}")
            if torch.cuda.is_available():
                used  = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"{'VRAM Used:':<20} {used:.1f}/{total:.1f} GB")

            plot_path = plot_epoch_progress(history, epoch, method, epoch_progress_dir)
            print(f"{'Progress plot:':<20} {os.path.basename(plot_path)}")

            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, history, best_acc, config, method
            )

            if val_acc > best_acc:
                best_acc   = val_acc
                model_path = config.get_model_path(method)
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc':            train_acc,
                    'val_acc':              val_acc,
                    'best_val_acc':         best_acc,
                    'train_loss':           train_loss,
                    'val_loss':             val_loss,
                }, model_path)
                print(f"{'Best model saved!':<20} Val Acc: {val_acc:.2f}%")

            print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED!".center(70))
        print("="*70)
        print(f"Progress saved. Resume dengan option [6] di menu.")
        print("="*70 + "\n")
        return history

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n\n" + "="*70)
            print("CUDA OUT OF MEMORY!".center(70))
            print("="*70)
            print(f"VRAM habis! Coba kurangi BATCH_SIZE di config.py:")
            current = getattr(config, 'BATCH_SIZE', 32)
            print(f"  Sekarang : BATCH_SIZE = {current}")
            print(f"  Coba     : BATCH_SIZE = {current // 2}")
            torch.cuda.empty_cache()
        else:
            print(f"\n\nError: {str(e)}")
            import traceback
            traceback.print_exc()
        return history

    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return history

    state_path = os.path.join(config.CHECKPOINTS_DIR, f'{method}_training_state.json')
    with open(state_path) as f:
        state = json.load(f)
    state['completed'] = True
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)

    print("\n" + "="*70)
    print("TRAINING COMPLETED!".center(70))
    print("="*70)
    print(f"{'Best Val Accuracy:':<30} {best_acc:.2f}%")
    print(f"{'Total Epochs:':<30} {config.NUM_EPOCHS}")
    print("="*70 + "\n")

    return history