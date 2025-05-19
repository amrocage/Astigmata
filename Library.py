import matplotlib.pyplot as plt 

def Plot_Bar(dataset_labels, dataset_sizes, title, xlabel, ylabel):
	plt.figure(figsize=(8,6))
	plt.bar(dataset_labels, dataset_sizes, color=["red", "green"])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, size in enumerate(dataset_sizes):
		plt.text(i, size + 100, str(size), ha="center", va="center")
	plt.show()

def Plot_ClassDistribution(dataset, dataset_name, class_names, title):
	class_counts = {}
	for _, label in dataset:
		class_name = class_names[label]
		class_counts[class_name] = class_counts.get(class_name, 0) + 1

	colors = ["skyblue", "lightgreen", "orange", "mediumorchid"]
	num_classes = len(class_names)

	plt.figure(figsize=(8,6))
	plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90, colors=colors[:num_classes])
	plt.title(f'Class Distribution in {dataset_name} Set')
	plt.axis("equal")
	plt.show();