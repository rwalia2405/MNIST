import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

#loading data

train_dataset = torchvision.datasets.MNIST(root = './data',download = True,train = True, transform = transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root = './data',train = False, transform = transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset = train_dataset, shuffle = True, batch_size = 50, num_workers = 2)
testloader = torch.utils.data.DataLoader(dataset = test_dataset, shuffle = False, batch_size = 50, num_workers = 2)

print("Loading of data Completed")

# visualising data
import matplotlib.pyplot as plt
dataiter = iter(trainloader)
image, label = dataiter.next()
print(label[0])
plt.imshow(image[0].view(-1,28),cmap = 'Greys')
plt.show()

#determinig the shape of input image
print(image.shape)

#defining our model

class Network(nn.Module):
	"""docstring for Network"""
	def __init__(self):
		super(Network, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
		self.norm1 = nn.BatchNorm2d(16)
		self.norm2 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.fc1 = nn.Linear(7*7*32, 128)
		self.fc2 = nn.Linear(128, 10)
		self.dropout = nn.Dropout(p = 0.2)

	def forward(self,x):
		x = self.maxpool(self.relu(self.norm1(self.conv1(x))))
		x = self.maxpool(self.relu(self.norm2(self.conv2(x))))
		x = x.view(-1,7*7*32)
		x = self.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

#initilizing our model
net = Network()

#defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)

#training the model
num_epochs = 5
total_step = len(trainloader)
for epoch in range(num_epochs):
	for i,(images,labels) in enumerate(trainloader):
		output = net(images)
		loss = criterion(output,labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1)%100 == 0:
			print('Epoch[{}/{}], step[{}/{}],loss = {: .4f}'.format(epoch+1,num_epochs,i+1,total_step,loss.item()))

#evaluating our model
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in testloader:
    	output = net(images)
    	_,predicted = torch.max(output.data,dim = 1)
    	total += labels.size(0)
    	correct += (predicted==labels).sum().item() 

    print('Accuracy of model on 10000 test images = {}%'.format((correct/total)*100))	

#predicting with our model

dataiter = iter(testloader)
image, label = dataiter.next()
with torch.no_grad():
	output = net(image)
_,predict = torch.max(output.data,dim = 1)
print('predicted = {}'.format(predict[0]))
print('actual = {}'.format(label[0]))	