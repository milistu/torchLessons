# TRAIN THE CLASSIFIER USING RESNET 18

This solution for a classification problem uses RESNET 18 as a backbone. <br>
Here is how to load the pre-trained RESNET18 model, modify its architecture and adapt to the desired problem.

## First iteration
### Test 1
- Accuracy of the network on the 50000 train images: 72.53%
- Accuracy of the network on the 10000 test images: 70.38%

### Test 2
Changed the last two linear layers to ReLU, decreased dropout to 20%.
- Accuracy of the network on the 50000 train images: 74.26%
- Accuracy of the network on the 10000 test images: 72.41%
