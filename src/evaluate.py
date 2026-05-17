import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils import AverageMeter
from src.visualization import plot_comprehensive_metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import os
import json


def evaluate_model(model, dataloader, config, method, feature_extractor=None, verbose=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    losses = AverageMeter()
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc='Evaluating') if verbose else dataloader
        
        for images, labels in iterator:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Forward pass based on method
            if method == 'baseline':
                outputs = model(images)
            
            elif method == 'residual_spatial':
                residual_images = feature_extractor.extract_batch(images)
                outputs = model(residual_images)
            
            elif method == 'residual_dct':
                residual_images = feature_extractor['residual'].extract_batch(images)
                dct_images = feature_extractor['dct'].extract_batch(residual_images)
                outputs = model(dct_images)
            
            elif method == 'fusion':
                residual_images = feature_extractor['residual'].extract_batch(images)
                dct_images = feature_extractor['dct'].extract_batch(residual_images)
                outputs = model(images, residual_images, dct_images)
            
            # Loss
            loss = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))
            
            # Predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores)
    
    # Calculate metrics
    accuracy = 100. * np.sum(y_true == y_pred) / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    
    results = {
        'loss': losses.avg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }
    
    return results


def full_evaluation(model, dataloader, config, method, feature_extractor=None, 
                   save_visualizations=True):
    print("\n" + "="*70)
    print(f"EVALUATION - {method.upper()}".center(70))
    print("="*70)
    
    # Evaluate
    results = evaluate_model(model, dataloader, config, method, feature_extractor)
    
    # Print metrics
    print(f"\n{'PERFORMANCE METRICS':^70}")
    print("="*70)
    print(f"{'Loss:':<20} {results['loss']:>10.4f}")
    print(f"{'Accuracy:':<20} {results['accuracy']:>9.2f}%")
    print(f"{'Precision:':<20} {results['precision']:>10.4f}")
    print(f"{'Recall:':<20} {results['recall']:>10.4f}")
    print(f"{'F1-Score:':<20} {results['f1_score']:>10.4f}")
    print(f"{'AUC:':<20} {results['auc']:>10.4f}")
    print("="*70)
    
    # Classification report
    print(f"\n{'CLASSIFICATION REPORT':^70}")
    print("="*70)
    print(classification_report(
        results['y_true'], 
        results['y_pred'],
        target_names=['Real', 'Fake'],
        digits=4
    ))
    print("="*70)
    
    # Save visualizations
    if save_visualizations:
        print(f"\nGenerating visualizations...")
        
        auc_score = plot_comprehensive_metrics(
            results['y_true'],
            results['y_pred'],
            results['y_scores'],
            method,
            config.VISUALIZATIONS_DIR,
            config
        )
        
        results['auc'] = auc_score
    
    # Save metrics to JSON
    metrics_summary = {
        'method': method,
        'loss': float(results['loss']),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'auc': float(results['auc'])
    }
    
    metrics_path = os.path.join(config.METRICS_DIR, f'{method}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save predictions
    if config.SAVE_PREDICTIONS:
        predictions_path = os.path.join(config.PREDICTIONS_DIR, f'{method}_predictions.npz')
        np.savez(
            predictions_path,
            y_true=results['y_true'],
            y_pred=results['y_pred'],
            y_scores=results['y_scores']
        )
        print(f"Predictions saved to: {predictions_path}")
    
    print("="*70 + "\n")
    
    return results


def load_and_evaluate(model_path, model, dataloader, config, method, feature_extractor=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    print(f"{'Checkpoint Information':^70}")
    print("="*70)
    print(f"{'Trained Epoch:':<25} {checkpoint.get('epoch', 'N/A')}")
    print(f"{'Training Accuracy:':<25} {checkpoint.get('train_acc', 0):.2f}%")
    print(f"{'Validation Accuracy:':<25} {checkpoint.get('val_acc', 0):.2f}%")
    print(f"{'Best Val Accuracy:':<25} {checkpoint.get('best_val_acc', 0):.2f}%")
    print("="*70)
    
    # Evaluate
    results = full_evaluation(model, dataloader, config, method, feature_extractor)
    
    return results


def evaluate_all_splits(model, train_loader, val_loader, test_loader, config, 
                        method, feature_extractor=None):
    print("\n" + "="*70)
    print(f"EVALUATING ON ALL SPLITS - {method.upper()}".center(70))
    print("="*70)
    
    all_results = {}
    
    # Evaluate on each split
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"\n{'â”€'*70}")
        print(f"{split_name.upper()} SET".center(70))
        print(f"{'â”€'*70}")
        
        results = evaluate_model(model, loader, config, method, feature_extractor, verbose=True)
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Accuracy:  {results['accuracy']:.2f}%")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  AUC:       {results['auc']:.4f}")
        
        all_results[split_name] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'auc': results['auc']
        }
    
    # Save combined results
    combined_path = os.path.join(config.METRICS_DIR, f'{method}_all_splits_metrics.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nCombined metrics saved to: {combined_path}")
    print("="*70 + "\n")
    
    return all_results