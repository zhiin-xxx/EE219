import os
import pickle
import numpy as np
from PIL import Image
import torchvision

def extract_cifar10(cifar_folder, output_dir):
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process train batches
    train_files = [f'data_batch_{i}' for i in range(1, 6)]
    train_label_file = open(os.path.join(train_dir, 'label.txt'), 'w')
    
    img_count = 0
    for file in train_files:
        file_path = os.path.join(cifar_folder, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            images = data['data']
            labels = data['labels']
            
            for i in range(len(images)):
                # Reshape and transpose to (32,32,3)
                img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img)
                
                img_name = f'train_{img_count}.png'
                img_path = os.path.join(train_dir, img_name)
                img.save(img_path)
                
                # Write label info
                label = labels[i]
                train_label_file.write(f'{img_name} {label} {classes[label]}\n')
                img_count += 1
    
    train_label_file.close()
    print(f'Train set done, {img_count} images')
    
    # Process test batch
    test_file = 'test_batch'
    test_file_path = os.path.join(cifar_folder, test_file)
    test_label_file = open(os.path.join(test_dir, 'label.txt'), 'w')
    
    img_count = 0
    with open(test_file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        images = data['data']
        labels = data['labels']
        
        for i in range(len(images)):
            # Reshape and transpose to (32,32,3)
            img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img)
            
            img_name = f'test_{img_count}.png'
            img_path = os.path.join(test_dir, img_name)
            img.save(img_path)
            
            # Write label info
            label = labels[i]
            test_label_file.write(f'{img_name} {label} {classes[label]}\n')
            img_count += 1
    
    test_label_file.close()
    print(f'Test set done, {img_count} images')
    print(f'Dataset saved to {output_dir}')

if __name__ == '__main__':
    # Download and extract CIFAR-10
    data_root = '/home/ubuntu/data/CIFAR10'
    os.makedirs(data_root, exist_ok=True)
    torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
    
    cifar_folder = os.path.join(data_root, 'cifar-10-batches-py')
    output_dir = os.path.join(data_root, 'datasets')
    
    extract_cifar10(cifar_folder, output_dir)
