import matplotlib.pyplot as plt
import re
import numpy as np

def plot_model_results(model_name, metrics):

    epochs = range(metrics['epochs'])

    plt.figure(figsize=(16, 6))
    plt.suptitle(f"Training and Validation Metrics for {model_name}")

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.legend()
    plt.ylim([0, max(max(metrics['loss']), max(metrics['val_loss']))])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')
    plt.title(f'Loss')


    plt.subplot(1, 2, 2)
    plt.plot(epochs, 100 * np.array(metrics['accuracy']), label='Train Accuracy')
    plt.plot(epochs, 100 * np.array(metrics['val_accuracy']), label='Validation Accuracy')
    plt.legend()
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.title(f'Accuracy')

    plt.show()

def get_top_models(results, metric="test_accuracy", top_n=5):

    sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)

    dict_model = {}

    for name, data in sorted_models[:top_n]:
        dict_model[name] = data[metric]

    return dict_model

def plot_metric_by_parameter(results, param_pattern, train=False):

    if param_pattern is None:
        filtered_models = results
    else:
        filtered_models = {name: data for name, data in results.items() if all(re.search(pattern, name) for pattern in param_pattern.split(' '))}

    if not filtered_models:
        print(f"No models match the pattern(s): {param_pattern}")
        return

    plt.figure(figsize=(10, 6))
    for model_name, data in filtered_models.items():
        plt.plot(data['val_accuracy'], label=f'{model_name} (val)', linestyle='-')
        if train:
            plt.plot(data['accuracy'], label=f'{model_name} (train)', linestyle='--')
    
    plt.title(f"Accuracy by parameter pattern: {param_pattern}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar_by_parameter(results, param_pattern=None):
    
    if param_pattern is None:
        filtered_models = results
    else:
        filtered_models = {name: data for name, data in results.items() if all(re.search(pattern, name) for pattern in param_pattern.split(' '))}

    if not filtered_models:
        print(f"No models match the pattern(s): {param_pattern}")
        return
    
    test_accuracies = []
    model_names = []
    for model_name, data in filtered_models.items():
        test_accuracies.append(data['test_accuracy'] * 100)
        model_names.append(model_name)

    plt.figure(figsize=(12, 6))
    y_pos = np.arange(len(model_names))
    
    plt.barh(y_pos, test_accuracies, align='center', color='skyblue')
    plt.yticks(y_pos, model_names)
    plt.xlabel('Accuracy (%)')
    plt.title(f"Validation Accuracy by Parameter Pattern: {param_pattern}")
    plt.xlim([0, 102])
    
    for i, v in enumerate(test_accuracies):
        plt.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()