import os

class Dataset_creator:

	def __init__(self, data):
		self.data = data

	def create_dataset(self):

		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		image_dir = os.path.join(BASE_DIR, self.data)
		i = 0
		y_labels = []
		x_train = []

		for root, dirs, files in os.walk(image_dir):
			for file in files:
				if file.endswith("png") or file.endswith("jpg"):
					path = os.path.join(root, file).replace("\\","/")
					label = os.path.basename(root)
					i+=1
					y_labels.append(label)
					x_train.append(path)

		print(i)
		print(x_train)
		return x_train,y_labels



if __name__ == '__main__':
    # create and show mainWindow
    mainWindow = Dataset_creator("Database1")
    retur = mainWindow.create_dataset()

