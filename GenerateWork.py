import os

import torch
from PIL import Image
from matplotlib import image
from matplotlib.pyplot import imshow, gcf
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import vgg19
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')


# Fixing the size of the image
imsize = 512
prep = transforms.Compose([transforms.Resize(imsize),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x:
                                             x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])

# Converting the generated image back to a format which we can visualise.
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])


def loadData():
    # Fixing the size of the image
    imsize = 512
    prep = transforms.Compose([transforms.Resize(imsize),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x:
                                                 x[torch.LongTensor([2,1,0])]),   # turn to BGR
                               transforms.Normalize(mean=[0.40760392,0.45795686,0.48501961],  # subtract imagenet mean
                                                    std = [1,1,1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    # Converting the generated image back to a format which we can visualise.
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                 transforms.Normalize(mean=[-0.40760392,-0.45795686,-0.48501961],std=[1,1,1]),
                                 transforms.Lambda(lambda x:x[torch.LongTensor([2,1,0])]),  # turn to BGR
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    # This method ensures data in the image does not cross the permissible range
    def postp(tensor):  # to clip results in the range [0,1]
        t = postpa(tensor)
        t[t > 1] = 1
        t[t < 0] = 0
        img = postpb(t)
        return img



    # A utility function to make a loading easier
    def image_Loader(image_name):
        image = Image.open(image_name)
        image = Variable(prep(image))
        # fake batch dimension required to fit network's input dimensions
        image = image.unsqueeze(0)
        return image
        pass

    style_img = image_Loader('/home/ZhangXueLiang/LiMiao/dataset/NeuralStyleTransfer/images/style.JPG')
    content_img = image_Loader('/home/ZhangXueLiang/LiMiao/dataset/NeuralStyleTransfer/images/conten.jpg')
    opt_img = Variable(content_img.data.clone(),requires_grad=True)


    pass

def postp(tensor):  # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img
# A utility function to make a loading easier
def image_Loader(image_name):
    image = Image.open(image_name)
    image = Variable(prep(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


def createModel():
    vgg = vgg19(pretrained = True).features

    # Freezing the layers as we will not use it for training
    for param in vgg.parameters():
        param.requires_grad = False

    return vgg


#  calculate the gram matrix
class GramMatrix(nn.Module):
    def forward(self,input):

        #  extracting the different dimensions from the input image
        b,c,h,w = input.size()

        #  flatten all the values along the height and width dimension
        features = input.view(b,c,h*w)

        # Calculate the gram matrix  by multiplying the flattening values along with its transposed vector
        gram_matrix = torch.bmm(features,features.transpose(1,2))
        gram_matrix.div_(h*w)
        return gram_matrix

#      calculate style loss
class StyleLoss(nn.Module):
    def forward(self,inputs,targets):
        out = nn.MSELoss()(GramMatrix()(inputs),targets)
        return (out)

class LayerActivations():
    features = []
    def __init__(self,model,layer_nums):
        self.hooks = []
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))

    def hook_fn(self,module,input,output):
        self.features.append(output)
    def remove(self):
        for hook in self.hooks:
            hook.remove()


def extract_layers(layers,img,model = None):
    la = LayerActivations(model,layers)
#     clearing the cache
    la.features = []
    out = model(img)
    la.remove()
    return la.features
def achieve():
    style_img = image_Loader('/home/ZhangXueLiang/LiMiao/dataset/NeuralStyleTransfer/images/style.JPG')
    content_img = image_Loader('/home/ZhangXueLiang/LiMiao/dataset/NeuralStyleTransfer/images/conten.jpg')

    vgg = createModel()
    style_img = style_img.cuda()
    content_img = content_img.cuda()
    vgg = vgg.cuda()
    opt_img = Variable(content_img.data.clone(), requires_grad=True)

    style_layers = [2,6,11,20,25]
    content_layers = [28]
    loss_layers = style_layers + content_layers

    content_targets = extract_layers(content_layers,content_img,model=vgg)
    content_targets = [t.detach() for t in content_targets]
    style_targets = extract_layers(style_layers,style_img,model=vgg)
    style_targets = [GramMatrix()(t).detach() for t in style_targets]
    targets = style_targets + content_targets

    # Creating loss function for each layers
    loss_fns = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    loss_fns = [fn.cuda() for fn in loss_fns]


    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights



    #
    # # calculate the MSE obtained from the outputs of these layers
    # target_layer = dummy_fn(content_img)
    # noise_layer = dummy_fn(opt_img)
    # criterion = nn.MSELoss()
    # content_loss = criterion(target_layer,noise_layer)
    #


    # Creating the optimizer
    optimizer = optim.LBFGS([opt_img])

    # training
    max_iter = 500
    show_iter = 50
    n_iter = [0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = extract_layers(loss_layers,opt_img,model=vgg)
            layer_losses = [weights[a] * loss_fns[a](A,targets[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
    #         print loss
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d,loss: %f'%(n_iter[0]+1,loss.item()))
            return loss
        optimizer.step(closure)

        out_img_hr = postp(opt_img.data[0].cpu().squeeze())
        imshow(out_img_hr)
        img = gcf().set_size_inches(10,10)
        plt.savefig('out_img_hr.jpg')
        plt.savefig('img.jpg')

    pass


def main():

    # print(f'Vgg:\n{createModel()}')
    # Image.open('/home/ZhangXueLiang/LiMiao/dataset/NeuralStyleTransfer/images/style.JPG').resize((600,600))
    achieve()


    pass


if __name__ == '__main__':
    main()