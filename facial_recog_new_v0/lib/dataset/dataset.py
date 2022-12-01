from torch.utils.data import Dataset
import cv2
import random

class FRDataset(Dataset):
	def __init__(self, image, label, transforms):
		self.image = image
		self.label = label
		self.transforms = transforms

	def __len__(self):
		return len(self.image)

	def __getitem__(self, idx):
		image = self.image[idx]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		label = self.label[idx]

		if self.transforms is not None:
			image = self.transforms(image)
			# image = self.transforms(image)

		return (image, label)
		# return (image, label)