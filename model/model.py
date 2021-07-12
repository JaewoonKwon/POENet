import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from utils.Dynamics import *
from utils.LieGroup import *
import torchvision.models as tviz_models


def makeLayers(nUnitList: list, layerType: str = 'Linear', activationType: str = 'Tanh'):
    # TODO: activationType
    assert 0 < len(nUnitList), f'len(nUnitList) = {len(nUnitList)}'
    layer = getattr(nn, layerType)
    activation = getattr(nn, activationType)
    jointLayer = nn.ModuleList([])
    for i in range(len(nUnitList) - 1):
        nIn = nUnitList[i]
        nOut = nUnitList[i + 1]
        jointLayer.append(layer(nIn, nOut))
        jointLayer.append(activation())
    # exclude the last activation
    del jointLayer[-1]
    return nn.Sequential(*jointLayer)


class POELayer(nn.Module):
    def __init__(self, nJoint, _useAdjoint):  # output dim = 6 (log se3)
        super(POELayer, self).__init__()
        # Adjoint representation
        _nominalTwist = revoluteTwist(torch.rand(nJoint, 6))
        # learnable parameter
        if _useAdjoint:
            self.eta = nn.Parameter(torch.Tensor(nJoint, 4))
        else:
            self.eta = nn.Parameter(torch.Tensor(nJoint, 6))
        self.M_se3 = nn.Parameter(torch.Tensor(1, 6))
        # initialize
        stdv = 1
        self.eta.data.uniform_(-stdv, stdv)
        self.M_se3.data.uniform_(-2, 2)
        # send it to the device
        self.initialM_se3 = self.M_se3.data
        self.register_buffer('nominalTwist', _nominalTwist)
        self.register_buffer('basis', self.getRevoluteBasis(_nominalTwist))
        self.register_buffer('useAdjoint', torch.tensor(_useAdjoint))

    def getRevoluteBasis(self, twist):
        nJoint = len(twist)
        basis = twist.new_empty(nJoint, 6, 4)  # eta is 4 dim
        perpendicular = twist.new_empty(2, 6)
        for i in range(nJoint):
            w = twist[i, :3]
            v = twist[i, 3:]
            perpendicular[0, :3] = w
            perpendicular[0, 3:] = v
            perpendicular[1, :3] = 0
            perpendicular[1, 3:] = w
            basis[i, :, :] = getNullspace(perpendicular)
        return basis

    def updateNominalTwist(self):
        if self.useAdjoint:
            self.nominalTwist = self.getJointTwist().detach()
            self.basis = self.getRevoluteBasis(self.nominalTwist).detach()
            self.eta.data.fill_(1e-20)
        else:
            pass

    def getJointTwist(self):
        if self.useAdjoint:
            nJoint = len(self.nominalTwist)
            expEta = expSE3(self.basis @ self.eta.view(nJoint, 4, 1))  # (nBatch,4,4)
            jointTwist = (largeAdjoint(expEta) @ self.nominalTwist.view(nJoint, 6, 1)).reshape(nJoint, 6)
            return jointTwist
        else:
            return self.eta

    def forward(self, x):
        # TODO : return 6-dim. se3 vector (Computing the loss, it should be applied to exp again)
        # return log_se3( exp^(A1 x1) * ... * exp^(An xn) )
        nBatch, nJoint = x.shape
        jointTwist = self.getJointTwist()
        if nJoint != jointTwist.shape[0]:
            print(f'[ERROR] POELayer.forward: jointTwist.shape = {jointTwist.shape}, x.shape = {x.shape}')
            exit(1)
        SE3 = x.new_zeros(nBatch, 4, 4)
        SE3[:, 0, 0] = SE3[:, 1, 1] = SE3[:, 2, 2] = SE3[:, 3, 3] = 1
        for i in range(nJoint):
            SE3 = SE3 @ expSE3(x[:, i].view(nBatch, 1) * jointTwist[i, :])
        SE3 = SE3 @ expSE3(self.M_se3).repeat(nBatch, 1, 1)
        return SE3


class AmbidexWristModel(BaseModel):
    def __init__(self, nJoint, useAdjoint):
        super(AmbidexWristModel, self).__init__()
        # an affine operation: y = Wx + b
        hiddenUnits = [2, 8, 32, 32, 16]
        # Joint angle layers
        self.nJoint = nJoint
        self.jointLayer = nn.ModuleList([])
        for i in range(len(hiddenUnits)):
            nIn = hiddenUnits[i]
            if i < len(hiddenUnits) - 1:
                nOut = hiddenUnits[i + 1]
                self.jointLayer.append(nn.Linear(nIn, nOut))
            else:
                nOut = nJoint
                self.jointLayer.append(nn.Linear(nIn, nOut))

        # POE layer
        self.poe = POELayer(nJoint, useAdjoint)

    def getJacobian(self, motorPos):
        nBatch = motorPos.shape[0]
        x = motorPos.unsqueeze(1)  # (nBatch, 1, nMotor)
        x = x.repeat(1, self.nJoint, 1)  # (nBatch, nJoint, nMotor)
        x.requires_grad_(True)
        output_val = torch.eye(self.nJoint).to(motorPos).reshape(1, self.nJoint, self.nJoint).repeat(nBatch, 1, 1)  # (nBatch, nJoint, nJoint)
        self.getJointAngle(x).backward(output_val)
        return x.grad.data  # (nBatch, nJoint, nMotor)

    def getJointStates(self, motorState: State):
        motorPos, motorVel, motorAcc = motorState.pos, motorState.vel, motorState.acc
        # jointPos
        jointPos = self.getJointAngle(motorPos)
        # jointVel
        getJointVel = lambda x, v: torch.autograd.functional.jvp(self.getJointAngle, x, v, create_graph=True, strict=False)[1]
        jointVel = getJointVel(motorPos, motorVel)
        # jointAcc
        J_acc = torch.autograd.functional.jvp(self.getJointAngle, motorPos, motorAcc, create_graph=True, strict=False)[1]
        dJdq_vel = torch.autograd.functional.jvp(lambda x: getJointVel(x, motorVel), motorPos, motorVel, create_graph=False, strict=False)[1]
        jointAcc = J_acc + dJdq_vel
        return State(jointPos, jointVel, jointAcc)

    def getJointAngle(self, x):
        for i in range(len(self.jointLayer) - 1):
            x = torch.tanh(self.jointLayer[i](x))
            # x = torch.nn.ReLU()(self.jointLayer[i](x))
        x = self.jointLayer[-1](x)
        return x

    def forward(self, x):
        x = self.getJointAngle(x)
        x = self.poe(x)
        return x


################################################ Euler Spiral

class EulerSpiralModel(BaseModel):
    def __init__(self, nJoint, useAdjoint):
        super(EulerSpiralModel, self).__init__()
        # an affine operation: y = Wx + b
        hiddenUnits = [1, 8, 32, 32, 16]
        # Joint angle layers
        self.jointLayer = nn.ModuleList([])
        for i in range(len(hiddenUnits)):
            nIn = hiddenUnits[i]
            if i < len(hiddenUnits) - 1:
                nOut = hiddenUnits[i + 1]
                self.jointLayer.append(nn.Linear(nIn, nOut))
            else:
                nOut = nJoint
                self.jointLayer.append(nn.Linear(nIn, nOut))
        # POE layer
        self.poe = POELayer(nJoint, useAdjoint)

    def getJointAngle(self, x):
        for i in range(len(self.jointLayer) - 1):
            x = torch.tanh(self.jointLayer[i](x))
            # x = torch.nn.ReLU()(self.jointLayer[i](x))
        x = self.jointLayer[-1](x)
        return x

    def forward(self, x):
        x = self.getJointAngle(x)
        x = self.poe(x)
        return x


################### ImageToPoseModel ###################

class ImageToPoseModel(BaseModel):
    def __init__(self, nJoint: int = 7, useAdjoint: bool = True, backbone: str = 'resnet101', pretrained: bool = True):
        super(ImageToPoseModel, self).__init__()
        if backbone == 'resnet18':
            net = tviz_models.resnet18(pretrained=pretrained)
            expansion = 1
        elif backbone == 'resnet34':
            net = tviz_models.resnet34(pretrained=pretrained)
            expansion = 1
        elif backbone == 'resnet50':
            net = tviz_models.resnet50(pretrained=pretrained)
            expansion = 4
        elif backbone == 'resnet101':
            net = tviz_models.resnet101(pretrained=pretrained)
            expansion = 4
        elif backbone == 'resnet152':
            net = tviz_models.resnet152(pretrained=pretrained)
            expansion = 4
        else:
            assert False, 'Invalid name of architecture.'
        # backbone
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        nFeature = 512 * expansion
        # Camera-to-Robot Pose (Camera frame w.r.t. robot)
        nPosition = [nFeature, 3]
        self.camposLayer = makeLayers(nPosition, layerType='Linear', activationType='Tanh')
        nQuat = [nFeature, 4]
        self.camquatLayer = makeLayers(nQuat, layerType='Linear', activationType='Tanh')
        # Joint angle
        nAngle = [nFeature, nJoint]
        self.angleLayer = makeLayers(nAngle, layerType='Linear', activationType='Tanh')
        # POE layer
        self.poe = POELayer(nJoint, useAdjoint)

    def getFeature(self, x):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def getCameraPose(self, x):
        feature = self.getFeature(x)
        campos = self.camposLayer(feature)
        camquat = self.camquatLayer(feature)
        camquat = F.normalize(camquat, p=2, dim=1)
        cameraPose = x.new_zeros(x.shape[0], 4, 4)
        cameraPose[:, 3, 3] = 1.0
        cameraPose[:, :3, 3] = campos
        cameraPose[:, :3, :3] = quaternions_to_rotation_matrices_torch(camquat)
        return cameraPose

    def getJointAngle(self, x):
        feature = self.getFeature(x)
        jointAngle = self.angleLayer(feature)
        return jointAngle

    def forward(self, x):
        # latent feature
        feature = self.getFeature(x)
        # Camera pose (Camera frame w.r.t. world)
        campos = self.camposLayer(feature)
        camquat = self.camquatLayer(feature)
        camquat = F.normalize(camquat, p=2, dim=1)
        cameraPose = x.new_zeros(x.shape[0], 4, 4)
        cameraPose[:, 3, 3] = 1.0
        cameraPose[:, :3, 3] = campos
        cameraPose[:, :3, :3] = quaternions_to_rotation_matrices_torch(camquat)
        # Joint angle
        jointAngle = self.angleLayer(feature)
        # End-effector pose
        # The POE layer has the joint screws expressed in the world frame.
        # EEpose is simply rotated by the camera pose
        # The world frame is ambiguous (but it's natural).
        # Dataset = pair of (Image, EE pose w.r.t camera frame)
        EEpose = cameraPose @ self.poe(jointAngle)
        return EEpose
