This is a project about Active Learning, here are the following instuctions:

* Load the Digit dataset from [here](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html),
* to renormalize the data, divide by 255,
* create a train and test set: the size of the test set should be 250;
* evaluate the performance of a [Logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) on the data set that you just created. **Remark:** do not change model's hyperparameters.

Split the train set above to create a pool set and a train set; the size of the new train set should be 200. **Remark:** if you use Scikit Learn's function train_test_split, do not use the option 'stratify'. At the end of this stage you should have obtained three data sets: a train set, a test set and a pool set.

The goal of this exercice is to implement the following learning algorithms:


1.   Random sampling,
2.   Active learning "least confidence" query strategy,
3.   Active learning "entropy" query strategy.

and to plot, for each of them, the learning curve, having the model accuracy and the iteration number on the y and x axis respectively.

The general form of these algorithms, written in pseudo-code, is given here below:

**input:** x_train, x_test, y_train, y_test, x_pool, y_pool, max_number_iterations,

**output** score_list, a list containing model's accuracy for each iteration.
```
for i < max_number_iterations:
  initialize a Logistic regression classifier clf
  learn clf on x_train, y_train
  get clf score on x_test, y_test
  append score to score_list
  *query an instance and its label from x_pool, y_pool
  append the queried instance and its label to x_train, y_train
  delete the queried instance and its label from x_pool, y_pool

```
where the line starting with * should be replaced by algorithms 1. 2. and 3. above. The parameter max_number_iterations can be set equal to 400.

You will find here below some hints to implement the three algorithms:

**Random sampling**: in this case you can just
* sample a random row from x_pool,
* get the corresponding label from y_pool,

**Least confidence**:
* get the predicted probability distribution y_prob for each sample in x_pool (by using clf.predict_proba() )
* compute the maximum probability max_proba of the matrix y_prob along each row (by using numpy.amax())
* get the vector of least confidence by doing 1 - max_proba
* by using numpy.argmax(), select the index associated with the sample which has least confidence,
* get the corresponding label from y_pool.

**Entropy**:
* get the predicted probability distribution y_prob for each sample in x_pool (by using clf.predict_proba(),
* compute the entropy along each row by using the formula -(y_prob * np.log2(y_prob)).sum(axis=1),
* by using numpy.argmax(), select the index of the row with maximum entropy,
* get the corresponding label from y_pool.

For each of the three sampling strategies, make a plot displaying iteration (x axis) vs accuracy (y axis) and comment the results.


What is the behavior of the three algorithms when varying the size of the training set and the number of iterations?
