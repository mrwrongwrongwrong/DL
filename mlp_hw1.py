import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.x = x
        output = x #in test1.py, x has 10 as batch size, 2 as dimension. consider 10 batches together as 1 epcho
        #linear1
        #print('x size:', x.size())
        W1 = self.parameters['W1']
        #print('W1 size:', W1.size())
        b1 = self.parameters['b1']
        rows = len(output) #rows of output, hence the dimension of number of input datapoints
        output =torch.transpose(output, 0, 1) #turning a 10*2 matrix to 2*10, preparing on feeding the input to next layer
        output = torch.matmul(W1,output) #W1*x has dimension 20*2 2*10, so 20*10, add b1 to each column
        rows = len(output) #get updated number of rows in current output
        columns = len(output[0]) # get number of columns of output, well output[0] is the first column

        b1 = torch.reshape(b1,(rows,1))
        expandedbias= b1.expand(rows,columns) #expand bias into the size as output
        output = output + expandedbias # add b1(10*1) to each column of the matrix 20*10
        z1 = output
        self.z1=z1

        if self.f_function == 'relu':#f_function
            output = torch.nn.ReLU()(output)
        elif self.f_function == 'sigmoid':
            output = torch.nn.Sigmoid()(output)
        elif self.f_function == 'identity':
            output = output
            
        self.z2 = output
        
        W2 = self.parameters['W2'] #torch.size[5*20]
        #print('W2 size:', W2.size())
        b2 = self.parameters['b2']#torch.size[5]
        #dimension of number of input datapoints, dimension of features
        columns = len(output[0])
        rows = len(output)

        output = torch.matmul(W2,output) #W1*x has dimension 5*20 20*10, so 5*10, add b1 to each column
        rows = len(output) #get updated number of rows in current output
        columns = len(output[0]) # get number of columns of output, well output[0] is the first column
        
        b2 = torch.reshape(b2,(rows,1))
        expandedbias= b2.expand(rows,columns) #expand bias into the size as output
        output = output + expandedbias # add b1(10*1) to each column of the matrix 20*10
        z3=output
        self.z3 = z3

        if self.g_function == 'relu':
            output = torch.nn.ReLU()(output)
        elif self.g_function == 'sigmoid':
            output = torch.nn.Sigmoid()(output)
        elif self.g_function == 'identity':
            output = output
        
        y_hat = output #row vector
        #print('yhat size before:',y_hat.size()) #5*10
        y_hat =torch.transpose(y_hat, 0, 1) #turning into column vector #10*5
        #print('yhat size after:',y_hat.size()) #10*5
        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        #get the value of dJdW1
        
        W2 = self.parameters['W2']
        W1 = self.parameters['W1']
        #print('dJdy_hat.size:',dJdy_hat.size()) #[10,5]

        if self.g_function == 'relu':
            dJdb2 = dJdy_hat * torch.transpose((self.z3 > 0), 0,1)
            
            dJdW2 = torch.matmul(self.z2, dJdb2) #20*10 10*5 ->20*5
            
        elif self.g_function == 'sigmoid':
            #torch.diagonal(dJdy_hat, 0) = 1
            #temp_jg= torch.nn.Sigmoid()(self.z3)
            temp_jg= torch.nn.Sigmoid()(self.z3)
            temp_jg= torch.mul(temp_jg, -1)
            temp_jg= torch.add(temp_jg, 1)
            #dJdb2 = dJdy_hat * torch.nn.Sigmoid()(z3) * temp_jg
            derivative_g = torch.mul(torch.nn.Sigmoid()(self.z3), temp_jg)#
            dJdb2 = dJdy_hat * torch.transpose(derivative_g, 0, 1)
            #dJdb2 = torch.matmul(dJdy_hat,dJdb2) #10*5 5*10
            dJdW2 = torch.matmul(self.z2, dJdb2) #20*10 10*5 ->20*5
            
        elif self.g_function == 'identity':
            #dJdb2 = dJdy_hat * W2
            #dJdb2 = dJdb2 *x
            dJdb2 = dJdy_hat #10*5
            #dJdb2 = torch.sum(dJdb2, dim = 1) #I summed up at the end where dataloader is
            #dJdW2 = dJdy_hat * torch.transpose(z2, 0, 1)
            dJdW2 = torch.matmul(self.z2, dJdy_hat)
        
        dJdW2 = torch.transpose(dJdW2, 0, 1) #5*20 #this is the transpose T mark
        
        layer2_out = torch.matmul(dJdb2, W2) #10*20
        #Super important!!
        layer2_out = torch.transpose(layer2_out, 0, 1) #20*10
        row_j = len(layer2_out)
        columns_j = len(layer2_out[0])
        
        #backward input[20,10] output weight 20*10 10* 2 ->[20,2]
        if self.f_function =='relu':

            dJdb1 = layer2_out * (self.z1 > 0)
            temp_dJdb1 = dJdb1
            dJdb1 = torch.sum(dJdb1, dim = 1) #[20,1]
            
            #print('temp_dJdb1',temp_dJdb1)
            #print('temp_dJdb1.size:',temp_dJdb1.size())
            dJdW1 = torch.matmul(temp_dJdb1, self.x)
            
        elif self.f_function =='sigmoid':
            temp_jg= torch.nn.Sigmoid()(self.z1) #z1: 20*10
            temp_jg= torch.mul(temp_jg, -1)
            temp_jg= torch.add(temp_jg, 1)
            #dJdb1 = layer2_out * torch.nn.Sigmoid()(z1) * temp_jg
            temp_f_derivative = torch.mul(torch.nn.Sigmoid()(self.z1),temp_jg) #20*10
            derivative_f = temp_f_derivative
            dJdb1 = layer2_out * derivative_f
            temp_dJdb1 = dJdb1
            dJdb1 = torch.sum(dJdb1, dim = 1)
            dJdW1 = torch.matmul(temp_dJdb1, self.x) #20*10 * 10*2
            
        elif self.f_function =='identity':
            #print('layer2_out.size',layer2_out.size())
            dJdb1 = layer2_out
            dJdb1 = torch.sum(dJdb1, dim = 1)
            dJdW1 = torch.matmul(layer2_out, self.x)
        self.grads['dJdW1'] = dJdW1
        #expected dJdW1 [20,2]
        self.grads['dJdb1'] = dJdb1
        #expected size of dJdb1 [20]
        self.grads['dJdW2'] = dJdW2
        #expected dJdW2 [5,20]
        self.grads['dJdb2'] = torch.sum(torch.transpose(dJdb2, 0, 1), dim = 1)
        #expected dJdb2 [5]
        
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    column_mse = len(y[0])
    row_mse = len(y)
    size = column_mse * row_mse
    temp = torch.sub(y, y_hat)
    temp = torch.square(temp)
    #wrong: temp = torch.sum(temp, dim=1)
    #correct:
    temp = torch.sum(temp)
    #wrong:
    #temp = torch.div(temp,column_mse)
    #loss = temp
    #correct:
    loss = temp /size
    
    #calculating dJdy_hat
    temp2 = torch.sub(y_hat, y)
    temp2 = torch.mul(temp2, 2)
    temp2 = temp2 /size
    dJdy_hat = temp2

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    column_bce = len(y[0])
    row_bce = len(y)
    size = column_bce * row_bce
    temp1_y = torch.mul(y,-1)
    temp1_y = torch.add(temp1_y,1)
    temp1_yhat = torch.mul(y_hat, -1)
    temp1_yhat = torch.add(temp1_yhat,1)
    temp1_yhat = torch.log(temp1_yhat)
    
    temp1_ylogyhat= torch.log(y_hat)
    temp1_ylogyhat = torch.mul(y,temp1_ylogyhat)
    temp1_beforesum = torch.mul(temp1_y,temp1_yhat)
    temp1_beforesum = torch.add(temp1_ylogyhat,temp1_beforesum)
    temp1_beforesum = torch.mul(temp1_beforesum, -1)
    loss = torch.sum(temp1_beforesum)/size
    
    
    print('bce loss',loss)
    
    temp_top = torch.sub(y_hat,y)
    temp_bot = torch.mul(y_hat, -1)
    temp_bot = torch.add(temp_bot,1)
    temp_bot = torch.mul(y_hat,temp_bot)
    dJdy_hat = torch.div(temp_top, temp_bot) /size
    print('BCE loss:',loss)
    return loss, dJdy_hat
