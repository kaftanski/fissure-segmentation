
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from constants import IMG_SHAPE
import numpy as np
import matplotlib.pyplot as plt
import argparse
import SimpleITK as sitk
import pickle
import os

from PIL import Image
from models3D.LPCA.model import *
from models3D.SmUtils.data import *
from models3D.SmUtils.metrics import *
from constants import COLORS_SHAPES
from mpl_toolkits.mplot3d import Axes3D

os.environ['MKL_NUM_THREADS'] = '1'
# variables:


def train( path, train_data, num_levels=3, target_variation=0.95, prefix="LPCA"):
    print('\n' + '###### LPCA Train ######' + '\n')
    model = LPCA(num_levels=num_levels, target_variation=target_variation)
    mean_vector, eigenvectors, eigenvalues, num_modes, percent_of_variance = model.lpca(train_data)
    with open(path + '/'+prefix+'_model.ssm', 'wb') as file:
        pickle.dump(model, file)
    txt = open(path + '/about_model.txt', 'w')
    txt.write('Model: ' + str(num_modes) + ' Modes \n'+
              'Variance: ' + str(percent_of_variance) + '\n'
              'Levels: ' + str(model.number_of_levels) + '\n'
              )
    txt.close()


def test(path, test_data, eval=False,prefix="LPCA"):
    with open(path + '/'+prefix+'_model.ssm', 'rb') as file:
        model_test = pickle.load(file)

    print('\n' + '###### LPCA Test ######' + '\n')

    projected_test_shapes= model_test.project_shapes(test_data)
    reconstructed_test_shapes = model_test.generate_shapes(projected_test_shapes)

    rec_loss=np.mean(np.abs(reconstructed_test_shapes-test_data))

    txt = open(path + '/about_model.txt', 'a')
    txt.write('\nRec Loss: ' + str(rec_loss))

    contours_rec = scr_datamatrix_to_contours(reconstructed_test_shapes)
    contours_real = scr_datamatrix_to_contours(test_data)
    for n in range(0,contours_rec[0].__len__(),30):
        plt.figure()
        G = grsp.GridSpec(1, 2)
        ax0 = plt.subplot(G[0, 0], projection='3d')
        ax1 = plt.subplot(G[0, 1], projection='3d')
        plot_shapes3D(test_data[:,n], ax0)
        plot_shapes3D(reconstructed_test_shapes[:,n],ax1)
        plt.savefig(path + '/rec_img'+str(n)+'.png')
    # Generate new Images:
    generate_imges(num_imgs=5, model_test=model_test, path=path)

    # Calculating Specifity:
    if eval:
        generalization = compute_generalization(model_test, test_data=test_data, metrics=["ASSD", "HD"])
        print(generalization)
        txt.write('\n\nGeneralization: ' + str(generalization))

        specifity = compute_specificity(model_test, train_data=train_data, test_data=test_data, metrics=["ASSD", "HD"])
        print(specifity)
        txt.write('\n\nSpecificity: ' + str(specifity))

    txt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_training", type=int, default=250, help="Number of training images")
    parser.add_argument("--n_folds", type=int, default=4, help="Number of folds")
    parser.add_argument("--n_levels", type=int, default=5, help="Number of locality levels")
    parser.add_argument("--eval", type=bool, default=False, help="Evaluate?")
    parser.add_argument("--test",type=bool, default=False, help="Test?")
    parser.add_argument("--train", type=bool, default=True, help="Train?")
    parser.add_argument("--var", type=float, default=0.95, help="Target variation")
    args = parser.parse_args()

    prefix="LPCA_nlvls_{}".format(args.n_levels)
    n_training = args.n_training - 10
    #save_path="/share/data_tiffy2/uzunova/PCA_AE_VAE/Results/"+prefix
    #save_path="/share/data_elmo1/ehrhardt/projects/ShapeModelComparison/2021_BVM/Results/"+prefix
    save_path="/share/data_wanda2/uzunova/PCA_AE_VAE/Results/3D/"+prefix
    for fold in range(args.n_folds):
        path = save_path+"/N"+str(args.n_training)+"/Fold"+str(fold)

        if not os.path.exists(path):
            os.makedirs(path)

        # load data and inits:
        train_data = get_ixi_point_data(mode="train", num_samples=n_training)
        train_data=align_GPA(train_data, num_iterations=5)
        test_data = get_ixi_point_data(mode ="test")
        test_data = align_GPA(test_data, num_iterations=5)

        if args.train:
            train(path, train_data, num_levels=args.n_levels, target_variation=args.var, prefix=prefix)
        if args.test:
            test(path, test_data, eval=args.eval, prefix=prefix)



