import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree, metrics
iris  = load_iris()
test_idx = [0,50,100]
########
	# print iris.feature_names
	# print iris.target_names
	# print iris.data[0]
	#print iris.target[1]

# for i in range(len(iris.target)):
# 	print "Example: %d: label %s, features %s" % (i, iris.target[i], iris.data[i])
########

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

pred = clf.predict(test_data)

acc = metrics.accuracy_score(test_target, pred)

# print test_target
print "accuracy:", acc

# viz code
# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(clf, 
# 	out_file=dot_data, 
# 	feature_names=iris.feature_names, 
# 	class_names=iris.target_names,
#     filled=True, 
# 	rounded=True,  
# 	impurity=False)
# graph = pydot.graph_from_dot_data(dot_data.getValue())
# graph.write_pdf("iris.pdf")
# import graphviz 
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                          feature_names=iris.feature_names,  
#                          class_names=iris.target_names,  
#                          filled=True, rounded=True,  
#                          special_characters=True) 
# graph = graphviz.Source(dot_data) 
# #graph.render("iris")
# graph.write_pdf("iris.pdf") 
