Mainly reproduction of Resnet ( [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) ), and also some other methods.

## Training

1. For training resnet:
` python -W ignore train.py --model=resnet --optim=mysgd --n=9 --batch_size=128 --gpus=0--lr=0.1 --n_epochs=200 --drop_p=0.0 --model_save=trained.pkl `

2. For training other models , follow the constructions in train.sh 

###### notice: The parameters after the last epoch is taken to be the final parameters since no remarkable overfitting observed.

## Testing

1. `python -W ignore test.py --test_mode --model_save=trained.pkl `

## Result

Implementation              | Cifar-10 Acc
----------------------------|-----------
Multi-dim Transformer      	|  91.41%
ResNet-56                   |  92.59%
Original ResNet-56          |  93.03%
