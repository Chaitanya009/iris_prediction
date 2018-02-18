from sklearn import tree,datasets,metrics,model_selection

iris = datasets.load_iris()

# print iris.DESCR

x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, random_state=42, test_size=0.33)


clf = tree.DecisionTreeClassifier().fit(x_train, y_train)

pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, pred)

print "accuracy:", acc
