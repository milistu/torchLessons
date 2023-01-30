### SAVE AND LOAD THE MODEL
import torch
import torchvision.models as models

# Download pretrained weights and save weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# Loading model wights
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''
NOTE:
    Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. 
    Failing to do this will yield inconsistent inference results.
'''
print(model.eval())

## Saving and Loading Models with Shapes
'''
    When loading model weights, we needed to instantiate the model class first, 
        because the class defines the structure of a network. 
    We might want to save the structure of this class together with the model, 
        in which case we can pass model (and not model.state_dict()) to the saving function:
'''
torch.save(model, 'model.pth')

# We can then load the model like this:
model = torch.load('model.pth')