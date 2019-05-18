import os
import argparse

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, \
    TensorBoard, LearningRateScheduler  # , EarlyStopping, ReduceLROnPlateau


from utils import get_model, get_cifar_gen, find_lr
from params import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name', default=model_name)
    parser.add_argument('--gpu', type=str, help='visiable devices', default=visible_gpu)
    parser.add_argument('--batch_size', type=int, help='batch size for dataloader', default=batch_size)
    parser.add_argument('--epochs', type=int, help='num epochs', default=epochs)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=lr)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = get_model(args.model)
    plot_model(model, to_file=os.path.join('experiments', args.model, args.model + '.png'), show_shapes=True)

    if not os.path.exists('experiments/' + args.model):
        os.makedirs(os.path.join('experiments', args.model, 'logs'))
        os.makedirs(os.path.join('experiments', args.model, 'checkpoints'))

    # callbacks
    model_names = os.path.join('experiments', args.model, 'checkpoints') + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', mode='min',
                                       verbose=1, save_best_only=True, 
                                       save_weights_only=False)
    # early_stop = EarlyStopping('val_loss', patience=patience)
    # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.2, patience=int(patience / 2), verbose=1)
    lr_scheduler = LearningRateScheduler(find_lr, verbose=1)
    tensor_board = TensorBoard(log_dir=os.path.join('experiments', args.model, 'logs'),
                               histogram_freq=0, write_graph=True, write_images=True)

    optimizer = SGD(lr=args.lr, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=optimizer, metrics=['accuracy'],
                  loss='categorical_crossentropy')

    cifar_gen, cifar_test_gen = get_cifar_gen()
    model.fit_generator(cifar_gen,
                        epochs=epochs,
                        validation_data=cifar_test_gen,
                        callbacks=[model_checkpoint,
                                   # early_stop, reduce_lr,
                                   tensor_board])
