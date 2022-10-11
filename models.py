import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F



class Perceptron(Module):
	def __init__(self, input_size=2, output_size=2, init_scale=.3, init_lr=.3):
		super().__init__()
		self.W = Parameter(init_scale * torch.rand([output_size, input_size]))
		self.b = Parameter(init_scale * torch.rand([output_size, 1]))
		self.lr_scalar = Parameter(torch.tensor([float(2 * init_lr)]))

	def forward(self, x):
		result = torch.matmul(self.W, x) + self.b
		result = nn.functional.sigmoid(result) * self.lr_scalar
		return result


class LRTPerceptron(Module):
	def __init__(self, input_size=2, output_size=2, regression=True, meta_learning=True, init_scale=.3):
		super().__init__()
		self.W = Parameter(init_scale * torch.rand([output_size, input_size]))
		self.b = Parameter(init_scale * torch.rand([output_size, 1]))
		self.W_lrn = Perceptron(input_size=input_size, output_size=input_size*output_size)
		self.b_lrn = Perceptron(input_size=input_size, output_size=output_size)
		self.optimizer = torch.optim.SGD([self.W, self.b], lr=0.01)
		self.lrn_optimizer = torch.optim.SGD(list(self.W_lrn.parameters()) + list(self.b_lrn.parameters()), lr=0.01)
		self.regression = regression
		self.meta_learning = meta_learning


	def forward(self, x):
		result = torch.matmul(self.W, x) + self.b
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def forward_temp(self, x, W_temp, b_temp):
		result = torch.matmul(W_temp, x) + b_temp
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def train_step(self, x, y, l, batch_size=50):
		self.optimizer.zero_grad()
		self.lrn_optimizer.zero_grad()
		loss = l(y, self.forward(x))
		loss.backward()
		if not self.meta_learning:
			self.optimizer.step()
			return 

		W_grad = torch.reshape(torch.clone(self.W.grad), [*self.W.shape, 1]).expand([*self.W.shape, batch_size])
		b_grad = torch.reshape(torch.clone(self.b.grad), [*self.b.shape, 1]).expand([*self.b.shape, batch_size])

		W_lrn_out = self.W_lrn(x)
		b_lrn_out = self.b_lrn(x)

		W_lrn_out = torch.reshape(W_lrn_out, [*self.W.shape, batch_size])
		b_lrn_out = torch.reshape(b_lrn_out, [*self.b.shape, batch_size])

		W_grad_full = W_lrn_out * W_grad
		b_grad_full = b_lrn_out * b_grad

		W_temp = self.W - torch.mean(W_grad_full, dim=2)
		b_temp = self.b - torch.mean(b_grad_full, dim=2)

		loss2 = l(y, self.forward_temp(x, W_temp, b_temp))

		loss2.backward()
		self.lrn_optimizer.step()
		
		with torch.no_grad():
			self.W -= torch.mean(W_grad_full, dim=2)
			self.b -= torch.mean(b_grad_full, dim=2)



class LRTMultilayerPerceptron(Module):
	def __init__(self, input_size=2, hidden_size=10, output_size=2, regression=True, meta_learning=True, init_scale=.1, lr=.0003):
		super().__init__()
		self.W1 = Parameter(init_scale * torch.rand([hidden_size, input_size]))
		self.b1 = Parameter(init_scale * torch.rand([hidden_size, 1]))
		self.W2 = Parameter(init_scale * torch.rand([output_size, hidden_size]))
		self.b2 = Parameter(init_scale * torch.rand([output_size, 1]))

		self.W1_lrn = Perceptron(input_size=input_size, output_size=input_size*hidden_size, init_scale=.03, init_lr=lr)
		self.b1_lrn = Perceptron(input_size=input_size, output_size=hidden_size, init_scale=.03, init_lr=lr)
		self.W2_lrn = Perceptron(input_size=input_size, output_size=hidden_size*output_size, init_scale=.03, init_lr=lr)
		self.b2_lrn = Perceptron(input_size=input_size, output_size=output_size, init_scale=.03, init_lr=lr)

		self.optimizer = torch.optim.SGD([self.W1, self.b1, self.W2, self.b2], lr=lr)
		self.lrn_optimizer = torch.optim.SGD(
			list(self.W1_lrn.parameters()) + 
			list(self.b1_lrn.parameters()) + 
			list(self.W2_lrn.parameters()) + 
			list(self.b2_lrn.parameters()), lr=lr)
		self.regression = regression
		self.meta_learning = meta_learning


	def forward(self, x):
		result = torch.matmul(self.W1, x) + self.b1
		result = torch.nn.functional.relu(result)
		result = torch.matmul(self.W2, result) + self.b2
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def forward_temp(self, x, W1_temp, b1_temp, W2_temp, b2_temp):
		result = torch.matmul(W1_temp, x) + b1_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W2_temp, result) + b2_temp
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def train_step(self, x, y, l, batch_size=50):
		self.optimizer.zero_grad()
		self.lrn_optimizer.zero_grad()
		loss = l(y, self.forward(x))
		loss.backward()
		if not self.meta_learning:
			self.optimizer.step()
			return 

		W1_grad = torch.reshape(torch.clone(self.W1.grad), [*self.W1.shape, 1]).expand([*self.W1.shape, batch_size])
		b1_grad = torch.reshape(torch.clone(self.b1.grad), [*self.b1.shape, 1]).expand([*self.b1.shape, batch_size])
		W2_grad = torch.reshape(torch.clone(self.W2.grad), [*self.W2.shape, 1]).expand([*self.W2.shape, batch_size])
		b2_grad = torch.reshape(torch.clone(self.b2.grad), [*self.b2.shape, 1]).expand([*self.b2.shape, batch_size])
		
		W1_lrn_out = self.W1_lrn(x)
		b1_lrn_out = self.b1_lrn(x)
		W2_lrn_out = self.W2_lrn(x)
		b2_lrn_out = self.b2_lrn(x)

		W1_lrn_out = torch.reshape(W1_lrn_out, [*self.W1.shape, batch_size])
		b1_lrn_out = torch.reshape(b1_lrn_out, [*self.b1.shape, batch_size])
		W2_lrn_out = torch.reshape(W2_lrn_out, [*self.W2.shape, batch_size])
		b2_lrn_out = torch.reshape(b2_lrn_out, [*self.b2.shape, batch_size])

		W1_grad_full = W1_lrn_out * W1_grad
		b1_grad_full = b1_lrn_out * b1_grad
		W2_grad_full = W2_lrn_out * W2_grad
		b2_grad_full = b2_lrn_out * b2_grad

		W1_temp = self.W1 - torch.mean(W1_grad_full, dim=2)
		b1_temp = self.b1 - torch.mean(b1_grad_full, dim=2)
		W2_temp = self.W2 - torch.mean(W2_grad_full, dim=2)
		b2_temp = self.b2 - torch.mean(b2_grad_full, dim=2)

		loss2 = l(y, self.forward_temp(x, W1_temp, b1_temp,  W2_temp, b2_temp))
		loss2.backward()
		self.lrn_optimizer.step()
		
		with torch.no_grad():
			self.W1 -= torch.mean(W1_grad_full, dim=2)
			self.b1 -= torch.mean(b1_grad_full, dim=2)
			self.W2 -= torch.mean(W2_grad_full, dim=2)
			self.b2 -= torch.mean(b2_grad_full, dim=2)


class LRTDeepMultilayerPerceptron(Module):
	def __init__(self, 
			input_size=2, 
			hidden_size1=10,  
			hidden_size2=10, 
			hidden_size3=10, 
			hidden_size4=10, 
			output_size=2, 
			regression=True, 
			meta_learning=True, 
			init_scale=.1, 
			lr=.0003):
		super().__init__()
		self.W1 = Parameter(init_scale * torch.rand([hidden_size1, input_size]))
		self.b1 = Parameter(init_scale * torch.rand([hidden_size1, 1]))
		self.W2 = Parameter(init_scale * torch.rand([hidden_size2, hidden_size1]))
		self.b2 = Parameter(init_scale * torch.rand([hidden_size2, 1]))
		self.W3 = Parameter(init_scale * torch.rand([hidden_size3, hidden_size2]))
		self.b3 = Parameter(init_scale * torch.rand([hidden_size3, 1]))
		self.W4 = Parameter(init_scale * torch.rand([hidden_size4, hidden_size4]))
		self.b4 = Parameter(init_scale * torch.rand([hidden_size4, 1]))
		self.W5 = Parameter(init_scale * torch.rand([output_size, hidden_size4]))
		self.b5 = Parameter(init_scale * torch.rand([output_size, 1]))

		self.W1_lrn = Perceptron(input_size=input_size, output_size=input_size*hidden_size1, init_scale=.03, init_lr=lr)
		self.b1_lrn = Perceptron(input_size=input_size, output_size=hidden_size1, init_scale=.03, init_lr=lr)
		self.W2_lrn = Perceptron(input_size=input_size, output_size=hidden_size1*hidden_size2, init_scale=.03, init_lr=lr)
		self.b2_lrn = Perceptron(input_size=input_size, output_size=hidden_size2, init_scale=.03, init_lr=lr)
		self.W3_lrn = Perceptron(input_size=input_size, output_size=hidden_size2*hidden_size3, init_scale=.03, init_lr=lr)
		self.b3_lrn = Perceptron(input_size=input_size, output_size=hidden_size3, init_scale=.03, init_lr=lr)
		self.W4_lrn = Perceptron(input_size=input_size, output_size=hidden_size3*hidden_size4, init_scale=.03, init_lr=lr)
		self.b4_lrn = Perceptron(input_size=input_size, output_size=hidden_size4, init_scale=.03, init_lr=lr)
		self.W5_lrn = Perceptron(input_size=input_size, output_size=hidden_size4*output_size, init_scale=.03, init_lr=lr)
		self.b5_lrn = Perceptron(input_size=input_size, output_size=output_size, init_scale=.03, init_lr=lr)

		self.optimizer = torch.optim.SGD([
			self.W1, 
			self.b1, 
			self.W2, 
			self.b2,
			self.W3, 
			self.b3,
			self.W4, 
			self.b4,
			self.W5, 
			self.b5], lr=lr)
		self.lrn_optimizer = torch.optim.SGD(
			list(self.W1_lrn.parameters()) + 
			list(self.b1_lrn.parameters()) + 
			list(self.W2_lrn.parameters()) + 
			list(self.b2_lrn.parameters()) + 
			list(self.W3_lrn.parameters()) + 
			list(self.b3_lrn.parameters()) + 
			list(self.W4_lrn.parameters()) + 
			list(self.b4_lrn.parameters()) + 
			list(self.W5_lrn.parameters()) + 
			list(self.b5_lrn.parameters()), lr=lr)
		self.regression = regression
		self.meta_learning = meta_learning


	def forward(self, x):
		result = torch.matmul(self.W1, x) + self.b1
		result = torch.nn.functional.relu(result)
		result = torch.matmul(self.W2, result) + self.b2
		result = torch.nn.functional.relu(result)
		result = torch.matmul(self.W3, result) + self.b3
		result = torch.nn.functional.relu(result)
		result = torch.matmul(self.W4, result) + self.b4
		result = torch.nn.functional.relu(result)
		result = torch.matmul(self.W5, result) + self.b5

		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def forward_temp(self, x, 
			W1_temp, b1_temp, 
			W2_temp, b2_temp,
			W3_temp, b3_temp,
			W4_temp, b4_temp,
			W5_temp, b5_temp):
		result = torch.matmul(W1_temp, x) + b1_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W2_temp, result) + b2_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W3_temp, result) + b3_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W4_temp, result) + b4_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W5_temp, result) + b5_temp
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def train_step(self, x, y, l, batch_size=50):
		self.optimizer.zero_grad()
		self.lrn_optimizer.zero_grad()
		loss = l(y, self.forward(x))
		loss.backward()
		if not self.meta_learning:
			self.optimizer.step()
			return 

		W1_grad = torch.reshape(torch.clone(self.W1.grad), [*self.W1.shape, 1]).expand([*self.W1.shape, batch_size])
		b1_grad = torch.reshape(torch.clone(self.b1.grad), [*self.b1.shape, 1]).expand([*self.b1.shape, batch_size])
		W2_grad = torch.reshape(torch.clone(self.W2.grad), [*self.W2.shape, 1]).expand([*self.W2.shape, batch_size])
		b2_grad = torch.reshape(torch.clone(self.b2.grad), [*self.b2.shape, 1]).expand([*self.b2.shape, batch_size])
		W3_grad = torch.reshape(torch.clone(self.W3.grad), [*self.W3.shape, 1]).expand([*self.W3.shape, batch_size])
		b3_grad = torch.reshape(torch.clone(self.b3.grad), [*self.b3.shape, 1]).expand([*self.b3.shape, batch_size])
		W4_grad = torch.reshape(torch.clone(self.W4.grad), [*self.W4.shape, 1]).expand([*self.W4.shape, batch_size])
		b4_grad = torch.reshape(torch.clone(self.b4.grad), [*self.b4.shape, 1]).expand([*self.b4.shape, batch_size])
		W5_grad = torch.reshape(torch.clone(self.W5.grad), [*self.W5.shape, 1]).expand([*self.W5.shape, batch_size])
		b5_grad = torch.reshape(torch.clone(self.b5.grad), [*self.b5.shape, 1]).expand([*self.b5.shape, batch_size])
		
		W1_lrn_out = self.W1_lrn(x)
		b1_lrn_out = self.b1_lrn(x)
		W2_lrn_out = self.W2_lrn(x)
		b2_lrn_out = self.b2_lrn(x)
		W3_lrn_out = self.W3_lrn(x)
		b3_lrn_out = self.b3_lrn(x)
		W4_lrn_out = self.W4_lrn(x)
		b4_lrn_out = self.b4_lrn(x)
		W5_lrn_out = self.W5_lrn(x)
		b5_lrn_out = self.b5_lrn(x)

		W1_lrn_out = torch.reshape(W1_lrn_out, [*self.W1.shape, batch_size])
		b1_lrn_out = torch.reshape(b1_lrn_out, [*self.b1.shape, batch_size])
		W2_lrn_out = torch.reshape(W2_lrn_out, [*self.W2.shape, batch_size])
		b2_lrn_out = torch.reshape(b2_lrn_out, [*self.b2.shape, batch_size])
		W3_lrn_out = torch.reshape(W3_lrn_out, [*self.W3.shape, batch_size])
		b3_lrn_out = torch.reshape(b3_lrn_out, [*self.b3.shape, batch_size])
		W4_lrn_out = torch.reshape(W4_lrn_out, [*self.W4.shape, batch_size])
		b4_lrn_out = torch.reshape(b4_lrn_out, [*self.b4.shape, batch_size])
		W5_lrn_out = torch.reshape(W5_lrn_out, [*self.W5.shape, batch_size])
		b5_lrn_out = torch.reshape(b5_lrn_out, [*self.b5.shape, batch_size])

		W1_grad_full = W1_lrn_out * W1_grad
		b1_grad_full = b1_lrn_out * b1_grad
		W2_grad_full = W2_lrn_out * W2_grad
		b2_grad_full = b2_lrn_out * b2_grad
		W3_grad_full = W3_lrn_out * W3_grad
		b3_grad_full = b3_lrn_out * b3_grad
		W4_grad_full = W4_lrn_out * W4_grad
		b4_grad_full = b4_lrn_out * b4_grad
		W5_grad_full = W5_lrn_out * W5_grad
		b5_grad_full = b5_lrn_out * b5_grad

		W1_temp = self.W1 - torch.mean(W1_grad_full, dim=2)
		b1_temp = self.b1 - torch.mean(b1_grad_full, dim=2)
		W2_temp = self.W2 - torch.mean(W2_grad_full, dim=2)
		b2_temp = self.b2 - torch.mean(b2_grad_full, dim=2)
		W3_temp = self.W3 - torch.mean(W3_grad_full, dim=2)
		b3_temp = self.b3 - torch.mean(b3_grad_full, dim=2)
		W4_temp = self.W4 - torch.mean(W4_grad_full, dim=2)
		b4_temp = self.b4 - torch.mean(b4_grad_full, dim=2)
		W5_temp = self.W5 - torch.mean(W5_grad_full, dim=2)
		b5_temp = self.b5 - torch.mean(b5_grad_full, dim=2)

		loss2 = l(y, self.forward_temp(x, 
			W1_temp, b1_temp,  
			W2_temp, b2_temp,  
			W3_temp, b3_temp,  
			W4_temp, b4_temp,  
			W5_temp, b5_temp))
		loss2.backward()
		self.lrn_optimizer.step()
		
		with torch.no_grad():
			self.W1 -= torch.mean(W1_grad_full, dim=2)
			self.b1 -= torch.mean(b1_grad_full, dim=2)
			self.W2 -= torch.mean(W2_grad_full, dim=2)
			self.b2 -= torch.mean(b2_grad_full, dim=2)
			self.W3 -= torch.mean(W3_grad_full, dim=2)
			self.b3 -= torch.mean(b3_grad_full, dim=2)
			self.W4 -= torch.mean(W4_grad_full, dim=2)
			self.b4 -= torch.mean(b4_grad_full, dim=2)
			self.W5 -= torch.mean(W5_grad_full, dim=2)
			self.b5 -= torch.mean(b5_grad_full, dim=2)


class HSLRTDeepMultilayerPerceptron(Module):
	def __init__(self, 
			input_size=2, 
			hidden_size1=10,  
			hidden_size2=10, 
			hidden_size3=10, 
			hidden_size4=10, 
			output_size=2, 
			regression=True, 
			meta_learning=True, 
			init_scale=.1, 
			lr=.0003):
		super().__init__()
		self.W1 = Parameter(init_scale * torch.rand([hidden_size1, input_size]))
		self.b1 = Parameter(init_scale * torch.rand([hidden_size1, 1]))
		self.W2 = Parameter(init_scale * torch.rand([hidden_size2, hidden_size1]))
		self.b2 = Parameter(init_scale * torch.rand([hidden_size2, 1]))
		self.W3 = Parameter(init_scale * torch.rand([hidden_size3, hidden_size2]))
		self.b3 = Parameter(init_scale * torch.rand([hidden_size3, 1]))
		self.W4 = Parameter(init_scale * torch.rand([hidden_size4, hidden_size4]))
		self.b4 = Parameter(init_scale * torch.rand([hidden_size4, 1]))
		self.W5 = Parameter(init_scale * torch.rand([output_size, hidden_size4]))
		self.b5 = Parameter(init_scale * torch.rand([output_size, 1]))

		self.W1_lrn = Perceptron(input_size=input_size, output_size=input_size*hidden_size1, init_scale=.03, init_lr=lr)
		self.b1_lrn = Perceptron(input_size=input_size, output_size=hidden_size1, init_scale=.03, init_lr=lr)
		self.W2_lrn = Perceptron(input_size=hidden_size1, output_size=hidden_size1*hidden_size2, init_scale=.03, init_lr=lr)
		self.b2_lrn = Perceptron(input_size=hidden_size1, output_size=hidden_size2, init_scale=.03, init_lr=lr)
		self.W3_lrn = Perceptron(input_size=hidden_size2, output_size=hidden_size2*hidden_size3, init_scale=.03, init_lr=lr)
		self.b3_lrn = Perceptron(input_size=hidden_size2, output_size=hidden_size3, init_scale=.03, init_lr=lr)
		self.W4_lrn = Perceptron(input_size=hidden_size3, output_size=hidden_size3*hidden_size4, init_scale=.03, init_lr=lr)
		self.b4_lrn = Perceptron(input_size=hidden_size3, output_size=hidden_size4, init_scale=.03, init_lr=lr)
		self.W5_lrn = Perceptron(input_size=hidden_size4, output_size=hidden_size4*output_size, init_scale=.03, init_lr=lr)
		self.b5_lrn = Perceptron(input_size=hidden_size4, output_size=output_size, init_scale=.03, init_lr=lr)

		self.optimizer = torch.optim.SGD([
			self.W1, 
			self.b1, 
			self.W2, 
			self.b2,
			self.W3, 
			self.b3,
			self.W4, 
			self.b4,
			self.W5, 
			self.b5], lr=lr)
		self.lrn_optimizer = torch.optim.SGD(
			list(self.W1_lrn.parameters()) + 
			list(self.b1_lrn.parameters()) + 
			list(self.W2_lrn.parameters()) + 
			list(self.b2_lrn.parameters()) + 
			list(self.W3_lrn.parameters()) + 
			list(self.b3_lrn.parameters()) + 
			list(self.W4_lrn.parameters()) + 
			list(self.b4_lrn.parameters()) + 
			list(self.W5_lrn.parameters()) + 
			list(self.b5_lrn.parameters()), lr=lr)
		self.regression = regression
		self.meta_learning = meta_learning


	def forward(self, x):
		result = torch.matmul(self.W1, x) + self.b1
		result = torch.nn.functional.relu(result)
		self._hidden_state1 = result
		result = torch.matmul(self.W2, result) + self.b2
		result = torch.nn.functional.relu(result)
		self._hidden_state2 = result
		result = torch.matmul(self.W3, result) + self.b3
		result = torch.nn.functional.relu(result)
		self._hidden_state3 = result
		result = torch.matmul(self.W4, result) + self.b4
		result = torch.nn.functional.relu(result)
		self._hidden_state4 = result
		result = torch.matmul(self.W5, result) + self.b5

		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def forward_temp(self, x, 
			W1_temp, b1_temp, 
			W2_temp, b2_temp,
			W3_temp, b3_temp,
			W4_temp, b4_temp,
			W5_temp, b5_temp):
		result = torch.matmul(W1_temp, x) + b1_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W2_temp, result) + b2_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W3_temp, result) + b3_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W4_temp, result) + b4_temp
		result = torch.nn.functional.relu(result)
		result = torch.matmul(W5_temp, result) + b5_temp
		if not self.regression:
			result = nn.functional.sigmoid(result)
		return result


	def train_step(self, x, y, l, batch_size=50):
		self.optimizer.zero_grad()
		self.lrn_optimizer.zero_grad()
		loss = l(y, self.forward(x))
		loss.backward()
		if not self.meta_learning:
			self.optimizer.step()
			return 

		with torch.no_grad():
			hidden_state1 = torch.clone(self._hidden_state1)
			hidden_state2 = torch.clone(self._hidden_state2)
			hidden_state3 = torch.clone(self._hidden_state3)
			hidden_state4 = torch.clone(self._hidden_state4)

		W1_grad = torch.reshape(torch.clone(self.W1.grad), [*self.W1.shape, 1]).expand([*self.W1.shape, batch_size])
		b1_grad = torch.reshape(torch.clone(self.b1.grad), [*self.b1.shape, 1]).expand([*self.b1.shape, batch_size])
		W2_grad = torch.reshape(torch.clone(self.W2.grad), [*self.W2.shape, 1]).expand([*self.W2.shape, batch_size])
		b2_grad = torch.reshape(torch.clone(self.b2.grad), [*self.b2.shape, 1]).expand([*self.b2.shape, batch_size])
		W3_grad = torch.reshape(torch.clone(self.W3.grad), [*self.W3.shape, 1]).expand([*self.W3.shape, batch_size])
		b3_grad = torch.reshape(torch.clone(self.b3.grad), [*self.b3.shape, 1]).expand([*self.b3.shape, batch_size])
		W4_grad = torch.reshape(torch.clone(self.W4.grad), [*self.W4.shape, 1]).expand([*self.W4.shape, batch_size])
		b4_grad = torch.reshape(torch.clone(self.b4.grad), [*self.b4.shape, 1]).expand([*self.b4.shape, batch_size])
		W5_grad = torch.reshape(torch.clone(self.W5.grad), [*self.W5.shape, 1]).expand([*self.W5.shape, batch_size])
		b5_grad = torch.reshape(torch.clone(self.b5.grad), [*self.b5.shape, 1]).expand([*self.b5.shape, batch_size])
		
		W1_lrn_out = self.W1_lrn(x)
		b1_lrn_out = self.b1_lrn(x)
		W2_lrn_out = self.W2_lrn(hidden_state1)
		b2_lrn_out = self.b2_lrn(hidden_state1)
		W3_lrn_out = self.W3_lrn(hidden_state2)
		b3_lrn_out = self.b3_lrn(hidden_state2)
		W4_lrn_out = self.W4_lrn(hidden_state3)
		b4_lrn_out = self.b4_lrn(hidden_state3)
		W5_lrn_out = self.W5_lrn(hidden_state4)
		b5_lrn_out = self.b5_lrn(hidden_state4)

		W1_lrn_out = torch.reshape(W1_lrn_out, [*self.W1.shape, batch_size])
		b1_lrn_out = torch.reshape(b1_lrn_out, [*self.b1.shape, batch_size])
		W2_lrn_out = torch.reshape(W2_lrn_out, [*self.W2.shape, batch_size])
		b2_lrn_out = torch.reshape(b2_lrn_out, [*self.b2.shape, batch_size])
		W3_lrn_out = torch.reshape(W3_lrn_out, [*self.W3.shape, batch_size])
		b3_lrn_out = torch.reshape(b3_lrn_out, [*self.b3.shape, batch_size])
		W4_lrn_out = torch.reshape(W4_lrn_out, [*self.W4.shape, batch_size])
		b4_lrn_out = torch.reshape(b4_lrn_out, [*self.b4.shape, batch_size])
		W5_lrn_out = torch.reshape(W5_lrn_out, [*self.W5.shape, batch_size])
		b5_lrn_out = torch.reshape(b5_lrn_out, [*self.b5.shape, batch_size])

		W1_grad_full = W1_lrn_out * W1_grad
		b1_grad_full = b1_lrn_out * b1_grad
		W2_grad_full = W2_lrn_out * W2_grad
		b2_grad_full = b2_lrn_out * b2_grad
		W3_grad_full = W3_lrn_out * W3_grad
		b3_grad_full = b3_lrn_out * b3_grad
		W4_grad_full = W4_lrn_out * W4_grad
		b4_grad_full = b4_lrn_out * b4_grad
		W5_grad_full = W5_lrn_out * W5_grad
		b5_grad_full = b5_lrn_out * b5_grad

		W1_temp = self.W1 - torch.mean(W1_grad_full, dim=2)
		b1_temp = self.b1 - torch.mean(b1_grad_full, dim=2)
		W2_temp = self.W2 - torch.mean(W2_grad_full, dim=2)
		b2_temp = self.b2 - torch.mean(b2_grad_full, dim=2)
		W3_temp = self.W3 - torch.mean(W3_grad_full, dim=2)
		b3_temp = self.b3 - torch.mean(b3_grad_full, dim=2)
		W4_temp = self.W4 - torch.mean(W4_grad_full, dim=2)
		b4_temp = self.b4 - torch.mean(b4_grad_full, dim=2)
		W5_temp = self.W5 - torch.mean(W5_grad_full, dim=2)
		b5_temp = self.b5 - torch.mean(b5_grad_full, dim=2)

		loss2 = l(y, self.forward_temp(x, 
			W1_temp, b1_temp,  
			W2_temp, b2_temp,  
			W3_temp, b3_temp,  
			W4_temp, b4_temp,  
			W5_temp, b5_temp))
		loss2.backward()
		self.lrn_optimizer.step()
		
		with torch.no_grad():
			self.W1 -= torch.mean(W1_grad_full, dim=2)
			self.b1 -= torch.mean(b1_grad_full, dim=2)
			self.W2 -= torch.mean(W2_grad_full, dim=2)
			self.b2 -= torch.mean(b2_grad_full, dim=2)
			self.W3 -= torch.mean(W3_grad_full, dim=2)
			self.b3 -= torch.mean(b3_grad_full, dim=2)
			self.W4 -= torch.mean(W4_grad_full, dim=2)
			self.b4 -= torch.mean(b4_grad_full, dim=2)
			self.W5 -= torch.mean(W5_grad_full, dim=2)
			self.b5 -= torch.mean(b5_grad_full, dim=2)

	class LRTResNet:
		pass


class ResNetModule(nn.Module):
	def __init__(self, channels, kernel_size=5):
		super(ResNetModule, self).__init__()
		if kernel_size % 2 == 0:
			raise "Only odd kernel sizes are supported."
		padding = (kernel_size - 1) // 2
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
		# self.conv1_W_LRT = Perceptron()
		# self.conv1_b_LRT = Perceptron()
		# We use batch norms to regularize.
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
		# self.conv2_W_LRT = Perceptron()
		# self.conv2_b_LRT = Perceptron()
		self.bn2 = nn.BatchNorm2d(channels)


	def forward(self, x):
		x2 = self.conv1(x)
		x2 = self.bn1(x2)
		x2 = F.relu(x2)
		x2 = self.conv2(x2)
		x2 = self.bn2(x2)
		x2 += x
		x2 = F.relu(x2)
		return x2

# The channel expansion module also optionally includes an initial pooling operation.
class ExpansionModule(nn.Module):
	def __init__(self, in_channels, out_channels, pool=True, kernel_size=5, padding=None):
		super(ExpansionModule, self).__init__()
		if kernel_size % 2 == 0:
			raise "Only odd kernel sizes are supported."
		if padding is None:
			padding = (kernel_size - 1) // 2
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		# self.conv1_W_LRT = Perceptron()
		# self.conv1_b_LRT = Perceptron()
		# We use batch norms to regularize.
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.pool = pool

	def forward(self, x):
		print(x.shape)
		if self.pool:
			x = F.max_pool2d(x, 2)
			print(x.shape)
		x = self.conv1(x)
		print(x.shape)
		x = self.bn1(x)
		print(x.shape)
		x = F.relu(x)
		return x


class ResNet10(nn.Module):
	def __init__(self, layer_depths=[1, 2, 2, 2], meta_learning=True, init_scale=.1, lr=.0003):
		super().__init__()
		channels = 32
		modules = [ExpansionModule(in_channels=3, out_channels=channels, pool=False, padding=0)]
		# for depth in layer_depths:
		# 	for i in range(depth):
		# 		modules.append(ResNetModule(channels=channels))
		# 	modules.append(ExpansionModule(in_channels=channels, out_channels=2*channels))
		# 	channels *= 2

		self.exp1 = ExpansionModule(in_channels=3, out_channels=channels, pool=False, padding=0)
		self.res1_1 = ResNetModule(channels=channels)
		self.exp2 = ExpansionModule(in_channels=channels, out_channels=2*channels)
		self.res2_1 = ResNetModule(channels=2*channels)
		self.res2_2 = ResNetModule(channels=2*channels)
		self.exp3 = ExpansionModule(in_channels=2*channels, out_channels=4*channels)
		self.res3_1 = ResNetModule(channels=4*channels)
		self.res3_2 = ResNetModule(channels=4*channels)
		self.exp4 = ExpansionModule(in_channels=4*channels, out_channels=8*channels)
		self.res4_1 = ResNetModule(channels=8*channels)
		self.res4_2 = ResNetModule(channels=8*channels)
		self.exp5 = ExpansionModule(in_channels=8*channels, out_channels=16*channels)
		self.fc1 = nn.Linear(512, 10)
		# self.fc1_W_LRT = Perceptron()
		# self.fc1_b_LRT = Perceptron()
		self.optimizer = torch.optim.SGD(
			list(self.exp1.parameters()) + 
			list(self.res1_1.parameters()) + 
			list(self.exp2.parameters()) +
			list(self.res2_1.parameters()) +
			list(self.res2_2.parameters()) + 
			list(self.exp3.parameters()) + 
			list(self.res3_1.parameters()) + 
			list(self.res3_2.parameters()) + 
			list(self.exp4.parameters()) + 
			list(self.res4_1.parameters()) + 
			list(self.res4_2.parameters()) + 
			list(self.exp5.parameters()), lr=lr)

		# self.lrn_optimizer = torch.optim.SGD(
		# 	list(self.exp1.conv1_W_LRT.parameters()) + 
		# 	list(self.exp1.conv1_b_LRT.parameters()) +
		# 	list(self.res1_1.conv1_W_LRT.parameters()) + 
		# 	list(self.res1_1.conv1_b_LRT.parameters()) +
		# 	list(self.res1_1.conv2_W_LRT.parameters()) + 
		# 	list(self.res1_1.conv2_b_LRT.parameters()) +
		# 	list(self.exp2.conv1_W_LRT.parameters()) + 
		# 	list(self.exp2.conv1_b_LRT.parameters()) +
		# 	list(self.res2_1.conv1_W_LRT.parameters()) + 
		# 	list(self.res2_1.conv1_b_LRT.parameters()) +
		# 	list(self.res2_1.conv2_W_LRT.parameters()) + 
		# 	list(self.res2_1.conv2_b_LRT.parameters()) +
		# 	list(self.res2_2.conv1_W_LRT.parameters()) + 
		# 	list(self.res2_2.conv1_b_LRT.parameters()) +
		# 	list(self.res2_2.conv2_W_LRT.parameters()) + 
		# 	list(self.res2_2.conv2_b_LRT.parameters()) + 
		# 	list(self.exp3.conv1_W_LRT.parameters()) + 
		# 	list(self.exp3.conv1_b_LRT.parameters()) +
		# 	list(self.res3_1.conv1_W_LRT.parameters()) + 
		# 	list(self.res3_1.conv1_b_LRT.parameters()) +
		# 	list(self.res3_1.conv2_W_LRT.parameters()) + 
		# 	list(self.res3_1.conv2_b_LRT.parameters()) +
		# 	list(self.res3_2.conv1_W_LRT.parameters()) + 
		# 	list(self.res3_2.conv1_b_LRT.parameters()) +
		# 	list(self.res3_2.conv2_W_LRT.parameters()) + 
		# 	list(self.res3_2.conv2_b_LRT.parameters()) + 
		# 	list(self.exp4.conv1_W_LRT.parameters()) + 
		# 	list(self.exp4.conv1_b_LRT.parameters()) +
		# 	list(self.res4_1.conv1_W_LRT.parameters()) + 
		# 	list(self.res4_1.conv1_b_LRT.parameters()) +
		# 	list(self.res4_1.conv2_W_LRT.parameters()) + 
		# 	list(self.res4_1.conv2_b_LRT.parameters()) +
		# 	list(self.res4_2.conv1_W_LRT.parameters()) + 
		# 	list(self.res4_2.conv1_b_LRT.parameters()) +
		# 	list(self.res4_2.conv2_W_LRT.parameters()) + 
		# 	list(self.res4_2.conv2_b_LRT.parameters()) + 
		# 	list(self.exp5.conv1_W_LRT.parameters()) + 
		# 	list(self.exp5.conv1_b_LRT.parameters()), lr=lr)
		
		# self.fc2 = nn.Linear(1000, nclasses)


	def forward(self, x):
		x = self.exp1(x)
		self._exp1_hidden = x
		print(self._exp1_hidden.shape)
		x = self.res1_1(x)
		self._res1_1_hidden = x
		print(self._res1_1_hidden.shape)
		x = self.exp2(x)
		self._exp2_hidden = x
		print(self._exp2_hidden.shape)
		x = self.res2_1(x)
		self._res2_1_hidden = x
		print(self._res2_1_hidden.shape)
		x = self.res2_2(x)
		self._res2_2_hidden = x
		print(self._res2_2_hidden.shape)
		x = self.exp3(x)
		self._exp3_hidden = x
		print(self._exp3_hidden.shape)
		x = self.res3_1(x)
		self._res3_1_hidden = x
		print(self._res3_1_hidden.shape)
		x = self.res3_2(x)
		self._res3_2_hidden = x
		print(self._res3_2_hidden.shape)
		x = self.exp4(x)
		self._exp4_hidden = x
		print(self._exp4_hidden.shape)
		x = self.res4_1(x)
		self._res4_1_hidden = x
		print(self._res4_1_hidden.shape)
		x = self.res4_2(x)
		self._res4_2_hidden = x
		print(self._res4_2_hidden.shape)
		x = self.exp5(x)
		self._exp5_hidden = x
		print(self._exp5_hidden.shape)

		x = x.view(-1, 512)
		x = self.fc1(x)
		# x = F.relu(x)
		# x = self.fc2(x)
		return F.log_softmax(x)

	def forward_temp(self, x,
			exp1_conv1_W_temp,
			exp1_conv1_b_temp,
			res1_1_conv1_W_temp,
			res1_1_conv1_b_temp,
			res1_1_conv2_W_temp,
			res1_1_conv2_b_temp,
			exp2_conv1_W_temp,
			exp2_conv1_b_temp,
			res2_1_conv1_W_temp,
			res2_1_conv1_b_temp,
			res2_1_conv2_W_temp,
			res2_1_conv2_b_temp,
			res2_2_conv1_W_temp,
			res2_2_conv1_b_temp,
			res2_2_conv2_W_temp,
			res2_2_conv2_b_temp,
			exp3_conv1_W_temp,
			exp3_conv1_b_temp,
			res3_1_conv1_W_temp,
			res3_1_conv1_b_temp,
			res3_1_conv2_W_temp,
			res3_1_conv2_b_temp,
			res3_2_conv1_W_temp,
			res3_2_conv1_b_temp,
			res3_2_conv2_W_temp,
			res3_2_conv2_b_temp,
			exp4_conv1_W_temp,
			exp4_conv1_b_temp,
			res4_1_conv1_W_temp,
			res4_1_conv1_b_temp,
			res4_1_conv2_W_temp,
			res4_1_conv2_b_temp,
			res4_2_conv1_W_temp,
			res4_2_conv1_b_temp,
			res4_2_conv2_W_temp,
			res4_2_conv2_b_temp,
			exp5_conv1_W_temp,
			exp5_conv1_b_temp,
			fc1_W_temp,
			fc1_b_temp):
		exp1 = ExpansionModule()
		exp1.conv1.weight = exp1_conv1_W_temp
		exp1.conv1.bias = exp1_conv1_b_temp
		res1_1 = ResNetModule()
		res1_1.conv1.weight = res1_1_conv1_W_temp
		res1_1.conv1.weight = res1_1_conv1_W_temp
		res1_1.conv2.weight = res1_1_conv2_W_temp
		res1_1.conv2.weight = res1_1_conv2_W_temp
		exp2 = ExpansionModule()
		exp2.conv1.weight = exp2_conv1_W_temp
		exp2.conv1.bias = exp2_conv1_b_temp
		res2_1 = ResNetModule()
		res2_1.conv1.weight = res2_1_conv1_W_temp
		res2_1.conv1.bias = res2_1_conv1_b_temp
		res2_1.conv2.weight = res2_1_conv2_W_temp
		res2_1.conv2.bias = res2_1_conv2_b_temp
		res2_2.conv1.weight = res2_2_conv1_W_temp
		res2_2.conv1.bias = res2_2_conv1_b_temp
		res2_2.conv2.weight = res2_2_conv2_W_temp
		res2_2.conv2.bias = res2_2_conv2_b_temp
		exp3 = ExpansionModule()
		exp3.conv1.weight = exp3_conv1_W_temp
		exp3.conv1.bias = exp3_conv1_b_temp
		res3_1 = ResNetModule()
		res3_1.conv1.weight = res3_1_conv1_W_temp
		res3_1.conv1.bias = res3_1_conv1_b_temp
		res3_1.conv2.weight = res3_1_conv2_W_temp
		res3_1.conv2.bias = res3_1_conv2_b_temp
		res3_2.conv1.weight = res3_2_conv1_W_temp
		res3_2.conv1.bias = res3_2_conv1_b_temp
		res3_2.conv2.weight = res3_2_conv2_W_temp
		res3_2.conv2.bias = res3_2_conv2_b_temp
		exp4 = ExpansionModule()
		exp4.conv1.weight = exp4_conv1_W_temp
		exp4.conv1.bias = exp4_conv1_b_temp
		res4_1 = ResNetModule()
		res4_1.conv1.weight = res4_1_conv1_W_temp
		res4_1.conv1.bias = res4_1_conv1_b_temp
		res4_1.conv2.weight = res4_1_conv2_W_temp
		res4_1.conv2.bias = res4_1_conv2_b_temp
		res4_2.conv1.weight = res4_2_conv1_W_temp
		res4_2.conv1.bias = res4_2_conv1_b_temp
		res4_2.conv2.weight = res4_2_conv2_W_temp
		res4_2.conv2.bias = res4_2_conv2_b_temp
		exp5 = ExpansionModule()
		exp5.conv1.weight = exp5_conv1_W_temp
		exp5.conv1.bias = exp5_conv1_b_temp

		x = self.exp1(x)
		x = self.res1_1(x)
		x = self.exp2(x)
		x = self.res2_1(x)
		x = self.res2_2(x)
		x = self.exp3(x)
		x = self.res3_1(x)
		x = self.res3_2(x)
		x = self.exp4(x)
		x = self.res4_1(x)
		x = self.res4_2(x)
		x = self.exp5(x)
		x = torch.matmul(fc1_W_temp, x) + fc1_b_temp
		return F.log_softmax(x)


	def train_step(self, x, y, l, batch_size=50):
		self.optimizer.zero_grad()
		self.lrn_optimizer.zero_grad()
		loss = l(y, self.forward(x))
		loss.backward()
		if not self.meta_learning:
			self.optimizer.step()
			return 

		exp1_conv1_W_grad =  torch.reshape(torch.clone(self.exp1.conv1.weight.grad), 
			[*self.exp1.conv1.weight.shape, 1]).expand([*self.exp1.conv1.weight.shape, batch_size])
		exp1_conv1_b_grad =  torch.reshape(torch.clone(self.exp1.conv1.bias.grad), 
			[*self.exp1.conv1.bias.shape, 1]).expand([*self.exp1.conv1.bias.shape, batch_size])
		
		res1_1_conv1_W_grad =  torch.reshape(torch.clone(self.res1_1.conv1.weight.grad), 
			[*self.res1_1.conv1.weight.shape, 1]).expand([*self.res1_1.conv1.weight.shape, batch_size])
		res1_1_conv1_b_grad =  torch.reshape(torch.clone(self.res1_1.conv1.bias.grad), 
			[*self.res1_1.conv1.bias.shape, 1]).expand([*self.res1_1.conv1.bias.shape, batch_size])
		res1_1_conv2_W_grad =  torch.reshape(torch.clone(self.res1_1.conv2.weight.grad), 
			[*self.res1_1.conv1.weight.shape, 1]).expand([*self.res1_1.conv2.weight.shape, batch_size])
		res1_1_conv2_b_grad =  torch.reshape(torch.clone(self.res1_1.conv2.bias.grad), 
			[*self.res1_1.conv2.bias.shape, 1]).expand([*self.res1_1.conv2.bias.shape, batch_size])

		exp2_conv1_W_grad =  torch.reshape(torch.clone(self.exp2.conv1.weight.grad), 
			[*self.exp2.conv1.weight.shape, 1]).expand([*self.exp2.conv1.weight.shape, batch_size])
		exp2_conv1_b_grad =  torch.reshape(torch.clone(self.exp2.conv1.bias.grad), 
			[*self.exp2.conv1.bias.shape, 1]).expand([*self.exp2.conv1.bias.shape, batch_size])

		res2_1_conv1_W_grad =  torch.reshape(torch.clone(self.res2_1.conv1.weight.grad), 
			[*self.res2_1.conv1.weight.shape, 1]).expand([*self.res2_1.conv1.weight.shape, batch_size])
		res2_1_conv1_b_grad =  torch.reshape(torch.clone(self.res2_1.conv1.bias.grad), 
			[*self.res2_1.conv1.bias.shape, 1]).expand([*self.res2_1.conv1.bias.shape, batch_size])
		res2_1_conv2_W_grad =  torch.reshape(torch.clone(self.res2_1.conv2.weight.grad), 
			[*self.res2_1.conv1.weight.shape, 1]).expand([*self.res2_1.conv2.weight.shape, batch_size])
		res2_1_conv2_b_grad =  torch.reshape(torch.clone(self.res2_1.conv2.bias.grad), 
			[*self.res2_1.conv2.bias.shape, 1]).expand([*self.res2_1.conv2.bias.shape, batch_size])

		res2_2_conv1_W_grad =  torch.reshape(torch.clone(self.res2_2.conv1.weight.grad), 
			[*self.res2_2.conv1.weight.shape, 1]).expand([*self.res2_2.conv1.weight.shape, batch_size])
		res2_2_conv1_b_grad =  torch.reshape(torch.clone(self.res2_2.conv1.bias.grad), 
			[*self.res2_2.conv1.bias.shape, 1]).expand([*self.res2_2.conv1.bias.shape, batch_size])
		res2_2_conv2_W_grad =  torch.reshape(torch.clone(self.res2_2.conv2.weight.grad), 
			[*self.res2_2.conv1.weight.shape, 1]).expand([*self.res2_2.conv2.weight.shape, batch_size])
		res2_2_conv2_b_grad =  torch.reshape(torch.clone(self.res2_2.conv2.bias.grad), 
			[*self.res2_2.conv2.bias.shape, 1]).expand([*self.res2_2.conv2.bias.shape, batch_size])

		exp3_conv1_W_grad =  torch.reshape(torch.clone(self.exp3.conv1.weight.grad), 
			[*self.exp3.conv1.weight.shape, 1]).expand([*self.exp3.conv1.weight.shape, batch_size])
		exp3_conv1_b_grad =  torch.reshape(torch.clone(self.exp3.conv1.bias.grad), 
			[*self.exp3.conv1.bias.shape, 1]).expand([*self.exp3.conv1.bias.shape, batch_size])

		res3_1_conv1_W_grad =  torch.reshape(torch.clone(self.res3_1.conv1.weight.grad), 
			[*self.res3_1.conv1.weight.shape, 1]).expand([*self.res3_1.conv1.weight.shape, batch_size])
		res3_1_conv1_b_grad =  torch.reshape(torch.clone(self.res3_1.conv1.bias.grad), 
			[*self.res3_1.conv1.bias.shape, 1]).expand([*self.res3_1.conv1.bias.shape, batch_size])
		res3_1_conv2_W_grad =  torch.reshape(torch.clone(self.res3_1.conv2.weight.grad), 
			[*self.res3_1.conv1.weight.shape, 1]).expand([*self.res3_1.conv2.weight.shape, batch_size])
		res3_1_conv2_b_grad =  torch.reshape(torch.clone(self.res3_1.conv2.bias.grad), 
			[*self.res3_1.conv2.bias.shape, 1]).expand([*self.res3_1.conv2.bias.shape, batch_size])

		res3_2_conv1_W_grad =  torch.reshape(torch.clone(self.res3_2.conv1.weight.grad), 
			[*self.res3_2.conv1.weight.shape, 1]).expand([*self.res3_2.conv1.weight.shape, batch_size])
		res3_2_conv1_b_grad =  torch.reshape(torch.clone(self.res3_2.conv1.bias.grad), 
			[*self.res3_2.conv1.bias.shape, 1]).expand([*self.res3_2.conv1.bias.shape, batch_size])
		res3_2_conv2_W_grad =  torch.reshape(torch.clone(self.res3_2.conv2.weight.grad), 
			[*self.res3_2.conv1.weight.shape, 1]).expand([*self.res3_2.conv2.weight.shape, batch_size])
		res3_2_conv2_b_grad =  torch.reshape(torch.clone(self.res3_2.conv2.bias.grad), 
			[*self.res3_2.conv2.bias.shape, 1]).expand([*self.res3_2.conv2.bias.shape, batch_size])

		exp4_conv1_W_grad =  torch.reshape(torch.clone(self.exp4.conv1.weight.grad), 
			[*self.exp4.conv1.weight.shape, 1]).expand([*self.exp4.conv1.weight.shape, batch_size])
		exp4_conv1_b_grad =  torch.reshape(torch.clone(self.exp4.conv1.bias.grad), 
			[*self.exp4.conv1.bias.shape, 1]).expand([*self.exp4.conv1.bias.shape, batch_size])

		res4_1_conv1_W_grad =  torch.reshape(torch.clone(self.res3_1.conv1.weight.grad), 
			[*self.res3_1.conv1.weight.shape, 1]).expand([*self.res3_1.conv1.weight.shape, batch_size])
		res4_1_conv1_b_grad =  torch.reshape(torch.clone(self.res3_1.conv1.bias.grad), 
			[*self.res3_1.conv1.bias.shape, 1]).expand([*self.res3_1.conv1.bias.shape, batch_size])
		res4_1_conv2_W_grad =  torch.reshape(torch.clone(self.res3_1.conv2.weight.grad), 
			[*self.res3_1.conv1.weight.shape, 1]).expand([*self.res3_1.conv2.weight.shape, batch_size])
		res4_1_conv2_b_grad =  torch.reshape(torch.clone(self.res3_1.conv2.bias.grad), 
			[*self.res3_1.conv2.bias.shape, 1]).expand([*self.res3_1.conv2.bias.shape, batch_size])

		res4_2_conv1_W_grad =  torch.reshape(torch.clone(self.res4_2.conv1.weight.grad), 
			[*self.res4_2.conv1.weight.shape, 1]).expand([*self.res4_2.conv1.weight.shape, batch_size])
		res4_2_conv1_b_grad =  torch.reshape(torch.clone(self.res4_2.conv1.bias.grad), 
			[*self.res4_2.conv1.bias.shape, 1]).expand([*self.res4_2.conv1.bias.shape, batch_size])
		res4_2_conv2_W_grad =  torch.reshape(torch.clone(self.res4_2.conv2.weight.grad), 
			[*self.res4_2.conv1.weight.shape, 1]).expand([*self.res4_2.conv2.weight.shape, batch_size])
		res4_2_conv2_b_grad =  torch.reshape(torch.clone(self.res4_2.conv2.bias.grad), 
			[*self.res4_2.conv2.bias.shape, 1]).expand([*self.res4_2.conv2.bias.shape, batch_size])

		exp5_conv1_W_grad =  torch.reshape(torch.clone(self.exp5.conv1.weight.grad), 
			[*self.exp5.conv1.weight.shape, 1]).expand([*self.exp5.conv1.weight.shape, batch_size])
		exp5_conv1_b_grad =  torch.reshape(torch.clone(self.exp5.conv1.bias.grad), 
			[*self.exp5.conv1.bias.shape, 1]).expand([*self.exp5.conv1.bias.shape, batch_size])

		fc1_W_grad = torch.reshape(torch.clone(self.fc1.weight.grad),
			[*self.fc1.weight.shape, 1]).expand([*self.fc1.weight.shape, batch_size])
		fc1_b_grad = torch.reshape(torch.clone(self.fc1.bias.grad),
			[*self.fc1.bias.shape, 1]).expand([*self.fc1.bias.shape, batch_size])



		exp1_conv1_W_out = torch.reshape(self.exp1.conv1_W_LRT(x), [*self.exp1.conv1.weight.shape, batch_size])
		exp1_conv1_b_out = torch.reshape(self.exp1.conv1_b_LRT(x), [*self.exp1.conv1.bias.shape, batch_size])
		res1_1_conv1_W_out = torch.reshape(self.res1_1.conv1_W_LRT(self._exp1_hidden), [*self.res1_1.conv1.weight.shape, batch_size])
		res1_1_conv1_b_out = torch.reshape(self.res1_1.conv1_b_LRT(self._exp1_hidden), [*self.res1_1.conv1.bias.shape, batch_size])
		res1_1_conv2_W_out = torch.reshape(self.res1_2.conv2_W_LRT(self._exp1_hidden), [*self.res1_1.conv2.weight.shape, batch_size])
		res1_1_conv2_b_out = torch.reshape(self.res1_2.conv2_b_LRT(self._exp1_hidden), [*self.res1_1.conv2.bias.shape, batch_size])
		exp2_conv1_W_out = torch.reshape(self.exp2.conv1_W_LRT(self._res1_1_hidden), [*self.exp2.conv1.weight.shape, batch_size])
		exp2_conv1_b_out = torch.reshape(self.exp2.conv1_b_LRT(self._res1_1_hidden), [*self.exp2.conv1.bias.shape, batch_size])
		res2_1_conv1_W_out = torch.reshape(self.res2_1.conv1_W_LRT(self._exp2_hidden), [*self.res2_1.conv1.weight.shape, batch_size])
		res2_1_conv1_b_out = torch.reshape(self.res2_1.conv1_b_LRT(self._exp2_hidden), [*self.res2_1.conv1.bias.shape, batch_size])
		res2_1_conv2_W_out = torch.reshape(self.res2_1.conv2_W_LRT(self._exp2_hidden), [*self.res2_1.conv2.weight.shape, batch_size])
		res2_1_conv2_b_out = torch.reshape(self.res2_1.conv2_b_LRT(self._exp2_hidden), [*self.res2_1.conv2.bias.shape, batch_size])
		res2_2_conv1_W_out = torch.reshape(self.res2_2.conv1_W_LRT(self._res2_1_hidden), [*self.res2_2.conv1.weight.shape, batch_size])
		res2_2_conv1_b_out = torch.reshape(self.res2_2.conv1_b_LRT(self._res2_1_hidden), [*self.res2_2.conv1.bias.shape, batch_size])
		res2_2_conv2_W_out = torch.reshape(self.res2_2.conv2_W_LRT(self._res2_1_hidden), [*self.res2_2.conv2.weight.shape, batch_size])
		res2_2_conv2_b_out = torch.reshape(self.res2_2.conv2_b_LRT(self._res2_1_hidden), [*self.res2_2.conv2.bias.shape, batch_size])
		exp3_conv1_W_out = torch.reshape(self.exp3.conv1_W_LRT(self._res2_1_hidden), [*self.exp3.conv1.weight.shape, batch_size])
		exp3_conv1_b_out = torch.reshape(self.exp3.conv1_b_LRT(self._res2_1_hidden), [*self.exp3.conv1.bias.shape, batch_size])
		res3_1_conv1_W_out = torch.reshape(self.res3_1.conv1_W_LRT(self._exp2_hidden), [*self.res3_1.conv1.weight.shape, batch_size])
		res3_1_conv1_b_out = torch.reshape(self.res3_1.conv1_b_LRT(self._exp2_hidden), [*self.res3_1.conv1.bias.shape, batch_size])
		res3_1_conv2_W_out = torch.reshape(self.res3_1.conv2_W_LRT(self._exp2_hidden), [*self.res3_1.conv2.weight.shape, batch_size])
		res3_1_conv2_b_out = torch.reshape(self.res3_1.conv2_b_LRT(self._exp2_hidden), [*self.res3_1.conv2.bias.shape, batch_size])
		res3_2_conv1_W_out = torch.reshape(self.res3_2.conv1_W_LRT(self._res3_1_hidden), [*self.res3_2.conv1.weight.shape, batch_size])
		res3_2_conv1_b_out = torch.reshape(self.res3_2.conv1_b_LRT(self._res3_1_hidden), [*self.res3_2.conv1.bias.shape, batch_size])
		res3_2_conv2_W_out = torch.reshape(self.res3_2.conv2_W_LRT(self._res3_1_hidden), [*self.res3_2.conv2.weight.shape, batch_size])
		res3_2_conv2_b_out = torch.reshape(self.res3_2.conv2_b_LRT(self._res3_1_hidden), [*self.res3_2.conv2.bias.shape, batch_size])
		exp4_conv1_W_out = torch.reshape(self.exp4.conv1_W_LRT(self._res3_2_hidden), [*self.exp3.conv1.weight.shape, batch_size])
		exp4_conv1_W_out = torch.reshape(self.exp4.conv1_b_LRT(self._res3_2_hidden), [*self.exp3.conv1.bias.shape, batch_size])
		res4_1_conv1_W_out = torch.reshape(self.res4_1.conv1_W_LRT(self._exp4_hidden), [*self.res4_1.conv1.weight.shape, batch_size])
		res4_1_conv1_b_out = torch.reshape(self.res4_1.conv1_b_LRT(self._exp4_hidden), [*self.res4_1.conv1.bias.shape, batch_size])
		res4_1_conv2_W_out = torch.reshape(self.res4_1.conv2_W_LRT(self._exp4_hidden), [*self.res4_1.conv2.weight.shape, batch_size])
		res4_1_conv2_b_out = torch.reshape(self.res4_1.conv2_b_LRT(self._exp4_hidden), [*self.res4_1.conv2.bias.shape, batch_size])
		res4_2_conv1_W_out = torch.reshape(self.res4_2.conv1_W_LRT(self._res4_1_hidden), [*self.res4_2.conv1.weight.shape, batch_size])
		res4_2_conv1_b_out = torch.reshape(self.res4_2.conv1_b_LRT(self._res4_1_hidden), [*self.res4_2.conv1.bias.shape, batch_size])
		res4_2_conv2_W_out = torch.reshape(self.res4_2.conv2_W_LRT(self._res4_1_hidden), [*self.res4_2.conv2.weight.shape, batch_size])
		res4_2_conv2_b_out = torch.reshape(self.res4_2.conv2_b_LRT(self._res4_1_hidden), [*self.res4_2.conv2.bias.shape, batch_size])
		exp5_conv1_W_out = torch.reshape(self.exp5.conv1_W_LRT(self._res4_2_hidden), [*self.exp5_1.conv1.weight.shape, batch_size])
		exp5_conv1_b_out = torch.reshape(self.exp5.conv1_b_LRT(self._res4_2_hidden), [*self.exp5_2.conv1.bias.shape, batch_size])
		fc1_W_out = torch.reshape(self.fc1_W_LRT(self._exp5_hidden), [self.fc1.weight.shape, batch_size])
		fc1_b_out = torch.reshape(self.fc1_b_LRT(self._exp5_hidden), [self.fc1.bias.shape, batch_size])


		exp1_conv1_W_grad_full = exp1_conv1_W_grad * exp1_conv1_W_out
		exp1_conv1_b_grad_full = exp1_conv1_b_grad * exp1_conv1_b_out
		res1_1_conv1_W_grad_full = res1_1_conv1_W_grad * res1_1_conv1_W_out
		res1_1_conv1_b_grad_full = res1_1_conv1_b_grad * res1_1_conv1_b_out
		res1_1_conv1_W_grad_full = res1_1_conv1_W_grad * res1_1_conv1_W_out
		res1_1_conv1_b_grad_full = res1_1_conv1_b_grad * res1_1_conv1_b_out
		exp2_conv1_W_grad_full = exp2_conv1_W_grad * exp2_conv1_W_out
		exp2_conv1_b_grad_full = exp2_conv1_b_grad * exp2_conv1_b_out
		res2_1_conv1_W_grad_full = res2_1_conv1_W_grad * res2_1_conv1_W_out
		res2_1_conv1_b_grad_full = res2_1_conv1_b_grad * res2_1_conv1_b_out
		res2_1_conv2_W_grad_full = res2_1_conv1_W_grad * res2_1_conv1_W_out
		res2_1_conv2_b_grad_full = res2_1_conv1_b_grad * res2_1_conv1_b_out
		res2_2_conv1_W_grad_full = res2_2_conv1_W_grad * res2_2_conv2_W_out
		res2_2_conv1_b_grad_full = res2_2_conv1_b_grad * res2_2_conv2_b_out
		res2_2_conv2_W_grad_full = res2_2_conv1_W_grad * res2_2_conv2_W_out
		res2_2_conv2_b_grad_full = res2_2_conv1_b_grad * res2_2_conv2_b_out
		exp3_conv1_W_grad_full = exp3_conv1_W_grad * exp3_conv1_W_out
		exp3_conv1_b_grad_full = exp3_conv1_b_grad * exp3_conv1_b_out
		res3_1_conv1_W_grad_full = res3_1_conv1_W_grad * res3_1_conv1_W_out
		res3_1_conv1_b_grad_full = res3_1_conv1_b_grad * res3_1_conv1_b_out
		res3_1_conv2_W_grad_full = res3_1_conv1_W_grad * res3_1_conv1_W_out
		res3_1_conv2_b_grad_full = res3_1_conv1_b_grad * res3_1_conv1_b_out
		res3_2_conv1_W_grad_full = res3_2_conv1_W_grad * res3_2_conv2_W_out
		res3_2_conv1_b_grad_full = res3_2_conv1_b_grad * res3_2_conv2_b_out
		res3_2_conv2_W_grad_full = res3_2_conv1_W_grad * res3_2_conv2_W_out
		res3_2_conv2_b_grad_full = res3_2_conv1_b_grad * res3_2_conv2_b_out
		exp4_conv1_W_grad_full = exp4_conv1_W_grad * exp4_conv1_W_out
		exp4_conv1_b_grad_full = exp4_conv1_b_grad * exp4_conv1_b_out
		res4_1_conv1_W_grad_full = res4_1_conv1_W_grad * res4_1_conv1_W_out
		res4_1_conv1_b_grad_full = res4_1_conv1_b_grad * res4_1_conv1_b_out
		res4_1_conv2_W_grad_full = res4_1_conv1_W_grad * res4_1_conv1_W_out
		res4_1_conv2_b_grad_full = res4_1_conv1_b_grad * res4_1_conv1_b_out
		res4_2_conv1_W_grad_full = res4_2_conv1_W_grad * res4_2_conv2_W_out
		res4_2_conv1_b_grad_full = res4_2_conv1_b_grad * res4_2_conv2_b_out
		res4_2_conv2_W_grad_full = res4_2_conv1_W_grad * res4_2_conv2_W_out
		res4_2_conv2_b_grad_full = res4_2_conv1_b_grad * res4_2_conv2_b_out
		exp5_conv1_W_grad_full = exp5_conv1_W_grad * exp5_conv1_W_out
		exp5_conv1_b_grad_full = exp5_conv1_b_grad * exp5_conv1_b_out
		fc1_W_grad_full = fc1_W_grad * fc1_W_out
		fc1_b_grad_full = fc1_b_grad * fc1_b_out

		exp1_conv1_W_temp = self.exp1.conv1.weight - torch.mean(exp1_conv1_W_grad_full, dim=2)
		exp1_conv1_b_temp = self.exp1.conv1.bias - torch.mean(exp1_conv1_b_grad_full, dim=2)
		res1_1_conv1_W_temp = self.res1_1.conv1.weight - torch.mean(res1_1_conv1_W_grad_full, dim=2)
		res1_1_conv1_b_temp = self.res1_1.conv1.bias - torch.mean(res1_1_conv1_b_grad_full, dim=2) 
		res1_1_conv1_W_temp = self.res1_1.conv2.weight - torch.mean(res1_1_conv1_W_grad_full, dim=2)
		res1_1_conv1_b_temp = self.res1_1.conv2.bias - torch.mean(res1_1_conv1_b_grad_full, dim=2)
		exp2_conv1_W_temp = self.exp1.conv1.weight - torch.mean(exp2_conv1_W_grad_full, dim=2)
		exp2_conv1_b_temp = self.exp1.conv1.bias - torch.mean(exp2_conv1_b_grad_full, dim=2)
		res2_1_conv1_W_temp = self.res2_1.conv1.weight - torch.mean(res2_1_conv1_W_grad_full, dim=2) 
		res2_1_conv1_b_temp = self.res2_1.conv1.bias - torch.mean(res2_1_conv1_b_grad_full, dim=2)
		res2_1_conv1_W_temp = self.res2_1.conv2.weight - torch.mean(res2_1_conv2_W_grad_full, dim=2)
		res2_1_conv1_b_temp = self.res2_1.conv2.bias - torch.mean(res2_1_conv2_b_grad_full, dim=2)
		res2_2_conv1_W_temp = self.res2_2.conv1.weight - torch.mean(res2_2_conv1_W_grad_full, dim=2)
		res2_2_conv1_b_temp = self.res2_2.conv1.bias - torch.mean(res2_2_conv1_b_grad_full, dim=2)
		res2_2_conv1_W_temp = self.res2_2.conv2.weight - torch.mean(res2_2_conv2_W_grad_full, dim=2)
		res2_2_conv1_b_temp = self.res2_2.conv2.bias - torch.mean(res2_2_conv2_b_grad_full, dim=2)
		exp3_conv1_W_temp = self.exp2.conv1.weight - torch.mean(exp3_conv1_W_grad_full, dim=2)
		exp3_conv1_b_temp = self.exp2.conv1.bias - torch.mean(exp3_conv1_b_grad_full, dim=2)
		res3_1_conv1_W_temp = self.res3_1.conv1.weight - torch.mean(res3_1_conv1_W_grad_full, dim=2)
		res3_1_conv1_b_temp = self.res3_1.conv1.bias - torch.mean(res3_1_conv1_b_grad_full, dim=2)
		res3_1_conv1_W_temp = self.res3_1.conv2.weight - torch.mean(res3_1_conv2_W_grad_full, dim=2)
		res3_1_conv1_b_temp = self.res3_1.conv2.bias - torch.mean(res3_1_conv2_b_grad_full, dim=2)
		res3_2_conv1_W_temp = self.res3_2.conv1.weight - torch.mean(res3_2_conv1_W_grad_full, dim=2)
		res3_2_conv1_b_temp = self.res3_2.conv1.bias - torch.mean(res3_2_conv1_b_grad_full, dim=2)
		res3_2_conv1_W_temp = self.res3_2.conv2.weight - torch.mean(res3_2_conv2_W_grad_full, dim=2)
		res3_2_conv1_b_temp = self.res3_2.conv2.bias - torch.mean(res3_2_conv2_b_grad_full, dim=2)
		exp4_conv1_W_temp = self.exp4.conv1.weight - torch.mean(exp4_conv1_W_grad_full, dim=2)
		exp4_conv1_b_temp = self.exp4.conv1.bias - torch.mean(exp4_conv1_b_grad_full, dim=2)
		res4_1_conv1_W_temp = self.res4_1.conv1.weight - torch.mean(res4_1_conv1_W_grad_full, dim=2)
		res4_1_conv1_b_temp = self.res4_1.conv1.bias - torch.mean(res4_1_conv1_b_grad_full, dim=2)
		res4_1_conv1_W_temp = self.res4_1.conv2.weight - torch.mean(res4_1_conv2_W_grad_full, dim=2)
		res4_1_conv1_b_temp = self.res4_1.conv2.bias - torch.mean(res4_1_conv2_b_grad_full, dim=2)
		res4_2_conv1_W_temp = self.res4_2.conv1.weight - torch.mean(res4_2_conv1_W_grad_full, dim=2)
		res4_2_conv1_b_temp = self.res4_2.conv1.bias - torch.mean(res4_2_conv1_b_grad_full, dim=2)
		res4_2_conv1_W_temp = self.res4_2.conv2.weight - torch.mean(res4_2_conv2_W_grad_full, dim=2)
		res4_2_conv1_b_temp = self.res4_2.conv2.bias - torch.mean(res4_2_conv2_b_grad_full, dim=2)
		exp5_conv1_W_temp = self.exp5.conv1.weight - torch.mean(exp5_conv1_W_grad_full, dim=2)
		exp5_conv1_b_temp = self.exp5.conv1.bias - torch.mean(exp5_conv1_b_grad_full, dim=2) 
		fc1_W_temp = self.fc1.weight - torch.mean(fc1_W_grad_full, dim=2)
		fc1_b_temp = self.fc1.bias = torch.mean(fc1_b_grad_full, dim=2)

		loss2 = l(y, self.forward_temp(x,
			exp1_conv1_W_temp,
			exp1_conv1_b_temp,
			res1_1_conv1_W_temp,
			res1_1_conv1_b_temp,
			res1_1_conv1_W_temp,
			res1_1_conv1_b_temp,
			exp2_conv1_W_temp,
			exp2_conv1_b_temp,
			res2_1_conv1_W_temp,
			res2_1_conv1_b_temp,
			res2_1_conv1_W_temp,
			res2_1_conv1_b_temp,
			res2_2_conv1_W_temp,
			res2_2_conv1_b_temp,
			res2_2_conv1_W_temp,
			res2_2_conv1_b_temp,
			exp3_conv1_W_temp,
			exp3_conv1_b_temp,
			res3_1_conv1_W_temp,
			res3_1_conv1_b_temp,
			res3_1_conv1_W_temp,
			res3_1_conv1_b_temp,
			res3_2_conv1_W_temp,
			res3_2_conv1_b_temp,
			res3_2_conv1_W_temp,
			res3_2_conv1_b_temp,
			exp4_conv1_W_temp,
			exp4_conv1_b_temp,
			res4_1_conv1_W_temp,
			res4_1_conv1_b_temp,
			res4_1_conv1_W_temp,
			res4_1_conv1_b_temp,
			res4_2_conv1_W_temp,
			res4_2_conv1_b_temp,
			res4_2_conv1_W_temp,
			res4_2_conv1_b_temp,
			exp5_conv1_W_temp,
			exp5_conv1_b_temp,
			fc1_W_temp,
			fc1_b_temp))
		
		loss2.backward()
		self.lrn_optimizer.step()
		with torch.no_grad():
			self.exp1.conv1.weight -= torch.mean(exp1_conv1_W_grad_full, dim=2)
			self.exp1.conv1.bias -= torch.mean(exp1_conv1_b_grad_full, dim=2)
			self.res1_1.conv1.weight -= torch.mean(res1_1_conv1_W_grad_full, dim=2)
			self.res1_1.conv1.bias -= torch.mean(res1_1_conv1_b_grad_full, dim=2) 
			self.res1_1.conv2.weight -= torch.mean(res1_1_conv1_W_grad_full, dim=2)
			self.res1_1.conv2.bias -= torch.mean(res1_1_conv1_b_grad_full, dim=2)
			self.exp1.conv1.weight -= torch.mean(exp2_conv1_W_grad_full, dim=2)
			self.exp1.conv1.bias -= torch.mean(exp2_conv1_b_grad_full, dim=2)
			self.res2_1.conv1.weight -= torch.mean(res2_1_conv1_W_grad_full, dim=2) 
			self.res2_1.conv1.bias -= torch.mean(res2_1_conv1_b_grad_full, dim=2)
			self.res2_1.conv2.weight -= torch.mean(res2_1_conv2_W_grad_full, dim=2)
			self.res2_1.conv2.bias -= torch.mean(res2_1_conv2_b_grad_full, dim=2)
			self.res2_2.conv1.weight -= torch.mean(res2_2_conv1_W_grad_full, dim=2)
			self.res2_2.conv1.bias -= torch.mean(res2_2_conv1_b_grad_full, dim=2)
			self.res2_2.conv2.weight -= torch.mean(res2_2_conv2_W_grad_full, dim=2)
			self.res2_2.conv2.bias -= torch.mean(res2_2_conv2_b_grad_full, dim=2)
			self.exp2.conv1.weight -= torch.mean(exp3_conv1_W_grad_full, dim=2)
			self.exp2.conv1.bias -= torch.mean(exp3_conv1_b_grad_full, dim=2)
			self.res3_1.conv1.weight -= torch.mean(res3_1_conv1_W_grad_full, dim=2)
			self.res3_1.conv1.bias -= torch.mean(res3_1_conv1_b_grad_full, dim=2)
			self.res3_1.conv2.weight -= torch.mean(res3_1_conv2_W_grad_full, dim=2)
			self.res3_1.conv2.bias -= torch.mean(res3_1_conv2_b_grad_full, dim=2)
			self.res3_2.conv1.weight -= torch.mean(res3_2_conv1_W_grad_full, dim=2)
			self.res3_2.conv1.bias -= torch.mean(res3_2_conv1_b_grad_full, dim=2)
			self.res3_2.conv2.weight -= torch.mean(res3_2_conv2_W_grad_full, dim=2)
			self.res3_2.conv2.bias -= torch.mean(res3_2_conv2_b_grad_full, dim=2)
			self.exp4.conv1.weight -= torch.mean(exp4_conv1_W_grad_full, dim=2)
			self.exp4.conv1.bias -= torch.mean(exp4_conv1_b_grad_full, dim=2)
			self.res4_1.conv1.weight -= torch.mean(res4_1_conv1_W_grad_full, dim=2)
			self.res4_1.conv1.bias -= torch.mean(res4_1_conv1_b_grad_full, dim=2)
			self.res4_1.conv2.weight -= torch.mean(res4_1_conv2_W_grad_full, dim=2)
			self.res4_1.conv2.bias -= torch.mean(res4_1_conv2_b_grad_full, dim=2)
			self.res4_2.conv1.weight -= torch.mean(res4_2_conv1_W_grad_full, dim=2)
			self.res4_2.conv1.bias -= torch.mean(res4_2_conv1_b_grad_full, dim=2)
			self.res4_2.conv2.weight -= torch.mean(res4_2_conv2_W_grad_full, dim=2)
			self.res4_2.conv2.bias -= torch.mean(res4_2_conv2_b_grad_full, dim=2)
			self.exp5.conv1.weight -= torch.mean(exp5_conv1_W_grad_full, dim=2)
			self.exp5.conv1.bias -= torch.mean(exp5_conv1_b_grad_full, dim=2) 
			self.fc1.weight -= torch.mean(fc1_W_grad_full, dim=2)
			self.fc1.bias -= torch.mean(fc1_b_grad_full, dim=2)


# TODO: Plot learning curve as function of # params?
# TODO: Resnet version
# TODO: Try many learning rates for baseline model
# TODO: Use standard initialization
# NOTE: Convolution module has 'weight' and 'bias' attributes

# MEASURE NORM OF LR TENSORS FOR REPEATED EXAMPLES, THEN NEW EXAMPLES. NORM SHOULD GO DOWN AS EXAMPLES REPEAT (OR ARE SIMILAR)
