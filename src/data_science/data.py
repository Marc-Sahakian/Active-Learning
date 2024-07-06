import numpy as np

def random_sampling(x_pool, y_pool):
    random_index = np.random.randint(0, len(x_pool))
    x_selected = x_pool[random_index]
    y_selected = y_pool[random_index]
    return x_selected, y_selected
def least_confidence(x_pool, y_pool, clf):
    confidences = clf.predict_proba(x_pool)
    least_confidence_index = np.argmin(np.max(confidences, axis=1))
    x_selected = x_pool[least_confidence_index]
    y_selected = y_pool[least_confidence_index]
    return x_selected, y_selected
def entropy(x_pool, y_pool, clf):
    entropies = -np.sum(clf.predict_proba(x_pool) * np.log2(clf.predict_proba(x_pool)), axis=1)
    highest_entropy_index = np.argmax(entropies)
    x_selected = x_pool[highest_entropy_index]
    y_selected = y_pool[highest_entropy_index]
    return x_selected, y_selected
def active_learning(x_train, y_train, x_test, y_test, x_pool, y_pool, max_number_iterations, query_strategy):
    score_list = []
    for i in range(max_number_iterations):
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        score = accuracy_score(y_test, clf.predict(x_test))
        score_list.append(score)
        
        if query_strategy == 'random':
            x_selected, y_selected = random_sampling(x_pool, y_pool)
        elif query_strategy == 'least_confidence':
            x_selected, y_selected = least_confidence(x_pool, y_pool, clf)
        elif query_strategy == 'entropy':
            x_selected, y_selected = entropy(x_pool, y_pool, clf)
        
        x_train = np.vstack([x_train, x_selected])
        y_train = np.append(y_train, y_selected)
        
        # Find index of selected instance in x_pool and remove it
        selected_index = np.where((x_pool == x_selected).all(axis=1))[0][0]
        x_pool = np.delete(x_pool, selected_index, axis=0)
        y_pool = np.delete(y_pool, selected_index)
        
    return score_list
