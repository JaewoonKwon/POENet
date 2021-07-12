from model.model import *
import data_loader.data_loaders as module_data

nJoint = 7
nBatch = 4
data_dir = 'data/panda'
model = ImageToPoseModel(nJoint=nJoint, backbone='resnet18', useAdjoint=True)
data_mean = [0.1377, 0.1984, 0.2584]
data_std = [0.1127, 0.1203, 0.1291]

eps = 1e-6


def test_image_dataloader():
    data_loader = module_data.ImageToPoseDataLoader(data_dir, nBatch, shuffle=True, validation_split=0.0, num_workers=1, training=True,data_mean=data_mean,data_std=data_std)
    for batch_idx, (images, poses) in enumerate(data_loader):
        batch_mean = images.mean(dim=(0,2,3))
        batch_std = images.std(dim=(0,2,3))
        threshold = 0.5
        assert torch.all((batch_mean - 0.0).abs()<threshold), f'MUST BE ZERO-MEAN: batch_mean={batch_mean}'
        assert torch.all((batch_std - 1.0).abs()<threshold), f'MUST BE STD=1: batch_std={batch_std}'

        imgBatch, rgbDim, height, width = images.shape
        assert imgBatch == nBatch, f'Check the image shape: imgBatch = {imgBatch}'
        assert rgbDim == 3, f'Check the image shape: rgbDim = {rgbDim}'
        assert height > 100, f'Check the image resolution: height = {height}'
        assert width > 100, f'Check the image resolution: width = {width}'
        assert poses.shape == (nBatch, 4, 4), f'poses.shape = {poses.shape}'

        cameraPose = model.getCameraPose(images)
        assert cameraPose.shape == (nBatch, 4, 4), f'cameraPose.shape = {cameraPose.shape}'
        jointAngle = model.getJointAngle(images)
        assert jointAngle.shape == (nBatch, nJoint), f'jointAngle.shape = {jointAngle.shape}'
        EEpose = model(images)
        assert EEpose.shape == (nBatch, 4, 4), f'EEpose.shape = {EEpose.shape}'

        if batch_idx > 3:
            break





