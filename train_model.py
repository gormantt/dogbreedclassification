#Author: Tim Gorman
#Date: 2023/01/18

#import libraries
import numpy as np
try:
    import boto3
except:
    pass
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import tempfile

try:
    from PIL import Image
except:
    pass

try:
    from smdebug import modes
    from smdebug.profiler.utils import str2bool
    from smdebug.pytorch import get_hook
except:
    pass
# The following class code is based largely on this solution posted in the following stack overflow link.

# https://stackoverflow.com/questions/54003052/how-do-i-implement-a-pytorch-dataset-for-use-with-aws-sagemaker

try:
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, bucket_name='dogbreedclassificationudacity', transform=None, phase='train'):
            self.bucket_name = bucket_name
            self.transform = transform
            self.phase = phase
            self.s3 = boto3.resource('s3')
            self.bucket = self.s3.Bucket(bucket_name)
            self.files = [obj.key for obj in self.bucket.objects.all() if obj.key.__contains__(phase)]   

            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_name = self.files[idx]

            # get image label
            label = torch.tensor(int(img_name.split('/')[1].split('.')[0])-1)

            # create temporary file name
            obj = self.bucket.Object(img_name)
            tmp = tempfile.NamedTemporaryFile()
            tmp_name = '{}.jpg'.format(tmp.name)

            # open tmp image
            with open(tmp_name, 'wb') as f:
                obj.download_fileobj(f)
                f.flush()
                f.close()
                image = Image.open(tmp_name)

            # apply transform to the image
            if self.transform:
                image = self.transform(image)

            return (image, label)
except:
    pass
    

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs):
    hook = get_hook(create_if_not_exists = True)
    if hook:
        hook.register_loss(criterion)
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                if hook:
                    hook.set_mode(modes.TRAIN)
                model.train()
            else:
                if hook:
                    hook.set_mode(modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


    
def create_data_loader(phase = 'train', batch_size = 100):
    if (phase == 'train') :
        return torch.utils.data.DataLoader(dataset = ImageDataset(phase = phase), batch_size = batch_size, shuffle = True)
    elif (phase == 'valid'):
        return torch.utils.data.DataLoader(dataset = ImageDataset(phase = phase), batch_size = batch_size, shuffle = True)
    else:
        return torch.utils.data.DataLoader(dataset = ImageDataset(phase = phase), batch_size = batch_size, shuffle = False)
            
def main():
    
    print('Defining Arguments')
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    args=parser.parse_args()
    
    # creating hook
    hook = get_hook(create_if_not_exists = True)
    print('Defining Device:')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Loading Model")
    model=net()
    
    # registering model
    if hook:
        hook.register_hook(model)
        
    print("Sending Model to Device")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    #path = 'resnet_retrained_model.pt'
    train_loader = create_data_loader(phase = 'train', batch_size = args.batch_size)
    validation_loader = create_data_loader(phase = 'valid', batch_size = args.batch_size)
    # test_loader = create_data_loader(phase = 'test', batch_size = args.test_batch_size)
    
    print('Training Model')
    model=train(model, train_loader, validation_loader, criterion, optimizer, device, args.epochs)
    
    torch.save(model.state_dict(), "/opt/ml/model/dogbreed_resnet50.pt")

if __name__=='__main__':
    
    main()