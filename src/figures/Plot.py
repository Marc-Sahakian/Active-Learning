import matplotlib.pyplot as plt

def plot_learning_curve_single(score_lists, max_number_iterations):
    """
    Plot learning curves for multiple strategies.
    
    Parameters:
    - score_lists: A dictionary where keys are strategy names and values are lists of scores.
    - max_number_iterations: The maximum number of iterations to plot.
    """
    plt.figure(figsize=(10, 6))
    for strategy, scores in score_lists.items():
        plt.plot(range(1, max_number_iterations + 1), scores, label=strategy.capitalize() + ' Sampling')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Active Learning Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_learning_curve_test_sizes(score_lists_dict, max_number_iterations):
    """
    Plot learning curves for different test sizes.
    
    Parameters:
    - score_lists_dict: A dictionary where keys are test sizes and values are dictionaries of score lists per strategy.
    - max_number_iterations: The maximum number of iterations to plot.
    """
    for test_size, score_lists in score_lists_dict.items():
        plt.figure(figsize=(10, 6))
        for strategy, scores in score_lists.items():
            plt.plot(range(1, max_number_iterations + 1), scores, label=strategy.capitalize() + ' Sampling')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Active Learning Learning Curves (Test Size: {test_size})')
        plt.legend()
        plt.grid(True)
        plt.show()

