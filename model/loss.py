import torch.nn.functional as F
from utils.Dynamics import *
from utils.LieGroup import *


def nll_loss(output, target):
    return F.nll_loss(output, target)


def SE3Error(output, target):
    nBatch = len(target)
    if output.shape != (nBatch, 4, 4) or target.shape != (nBatch, 4, 4):
        print(f'[ERROR] SE3Error : output.shape = {output.shape}, target.shape = {target.shape}')
        exit(1)
    errorse3 = skew_se3(logSE3(invSE3(output) @ target))  # (nBatch, 6)
    return errorse3.pow(2).sum() / nBatch


def unifiedError(output, target, alpha, jointState, motorState, Vdot0, twists, jacobian_theta):
    dynError = torqueError(jointState, motorState, Vdot0, twists, jacobian_theta)
    kineError = SE3Error(output, target)
    return dynError / alpha + kineError


def torqueError(jointState, motorState, Vdot0, twists, jacobian_theta):
    jointPos, jointVel, jointAcc = jointState.pos, jointState.vel, jointState.acc
    motorPos, motorVel, motorAcc, motorTorque = motorState.pos, motorState.vel, motorState.acc, motorState.torque

    nDynBatch = len(jointPos)
    nJoint = jointPos.shape[1]
    nMotor = motorTorque.shape[1]
    assert motorTorque.shape == (nDynBatch, nMotor), f'jointPos.shape = {jointPos.shape}, motorTorque.shape = {motorTorque.shape}'
    assert jacobian_theta.shape == (nDynBatch, nJoint, nMotor), f'jacobian_theta.shape = {jacobian_theta.shape}'
    Phi_dummy = jointPos.new_zeros(10 * nJoint)
    initialLinkFrames = torch.eye(4).to(jointPos).repeat(nJoint, 1, 1)
    F_ext = jointPos.new_zeros(6)
    # Y = (nDynBatch, nJoint, 10*nJoint)
    _, _, _, _, _, _, Y = solveRecursiveDynamics(jointPos, jointVel, jointAcc, Phi_dummy, twists, initialLinkFrames, Vdot0, F_ext)
    # jacobian_theta = (nDynBatch, nJoint, nMotor)
    # JtransY = (nDynBatch * nMotor, 10*nJoint)
    JtransY = (jacobian_theta.transpose(1, 2) @ Y).reshape(nDynBatch * nMotor, 10 * nJoint)
    # include motor friction and rotor inertia
    diagmotorvel = torch.diag_embed(motorVel, dim1=1, dim2=2).reshape(nDynBatch * nMotor, nMotor)  # (nDynBatch*nMotor, nMotor)
    diagmotoracc = torch.diag_embed(motorAcc, dim1=1, dim2=2).reshape(nDynBatch * nMotor, nMotor)  # (nDynBatch*nMotor, nMotor)
    JtransY_aug = torch.cat((JtransY, diagmotorvel.sign(), diagmotorvel, diagmotoracc), dim=1)
    # Truncated pseudo-inverse
    # linalg.pinv() is really BAD, e.g., pinvJtransY_aug = torch.linalg.pinv(JtransY)
    U, S, V = torch.svd(JtransY_aug)  # JtransY_aug = U@S.diag()@V.t()
    eps = 1e-4
    pinvJtransY_aug = V[:, S > eps] @ S[S > eps].pow(-1).diag() @ U[:, S > eps].t()
    # torque error
    Phi = pinvJtransY_aug @ motorTorque.reshape(nDynBatch * nMotor, )  # (10*nJoint, nDynBatch*nMotor) @ (nDynBatch*nMotor,)
    dynError = (motorTorque.reshape(nDynBatch * nMotor, ) - JtransY_aug @ Phi).pow(2).sum() / nDynBatch
    return dynError
