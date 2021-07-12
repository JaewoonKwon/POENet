from utils.LieGroup import *
from dataclasses import dataclass
import torch


@dataclass
class State:
    pos: torch.Tensor
    vel: torch.Tensor
    acc: torch.Tensor
    torque: torch.Tensor

    def __init__(self, pos, vel, acc, torque=None):
        assert pos.shape == vel.shape
        assert pos.shape == acc.shape
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.torque = torque


def solveKinematics(jointPos, jointVel, jointAcc, A_screw, initialLinkFrames, Vdot0):
    nBatch, nJoint = jointPos.shape
    if jointVel.shape != jointPos.shape or jointAcc.shape != jointPos.shape:
        print(f'[Error] solveRecursiveDynamics: jointVel.shape = {jointVel.shape}, jointAcc.shape = {jointAcc.shape}')
        exit(1)
    if Vdot0.shape != (6,) or A_screw.shape != (nJoint, 6) or initialLinkFrames.shape != (nJoint, 4, 4):
        print(f'[Error] solveRecursiveDynamics: Vdot0.shape = {Vdot0.shape}, A_screw.shape = {A_screw.shape}, initialLinkFrames.shape = {initialLinkFrames.shape}')
        exit(1)
    linkFrames = jointPos.new_zeros(nBatch, nJoint, 4, 4)
    V = jointPos.new_zeros(nBatch, nJoint, 6)
    Vdot = jointPos.new_zeros(nBatch, nJoint, 6)
    # Forward
    V_parent = V.new_zeros(nBatch, 6)
    Vdot_parent = Vdot0.repeat(nBatch, 1)  # (nBatch, 6)
    for i in range(nJoint):
        A_i = A_screw[i, :].repeat(nBatch, 1)  # (nBatch, 6)
        pos = jointPos[:, i].view(nBatch, 1)
        vel = jointVel[:, i].view(nBatch, 1)
        acc = jointAcc[:, i].view(nBatch, 1)
        # recursion
        T_i = initialLinkFrames[i] @ expSE3(A_i * pos)  # (nBatch, 4, 4)
        AdjInvT = largeAdjoint(invSE3(T_i))  # (nBatch, 6, 6)
        V_i = A_i * vel + (V_parent.reshape(nBatch, 1, 6) @ AdjInvT.transpose(1, 2)).squeeze()  # (nBatch, 6)
        Vdot_i = A_i * acc + (Vdot_parent.reshape(nBatch, 1, 6) @ AdjInvT.transpose(1, 2)).squeeze() \
                 + ((A_i * vel).reshape(nBatch, 1, 6) @ smallAdjoint(V_i).transpose(1, 2)).squeeze()  # (nBatch, 6)
        # log
        V_parent = V_i
        Vdot_parent = Vdot_i
        linkFrames[:, i, :, :] = T_i
        V[:, i, :] = V_i
        Vdot[:, i, :] = Vdot_i
    return V, Vdot, linkFrames


def solveRecursiveDynamics(jointPos, jointVel, jointAcc, Phi, A_screw, initialLinkFrames, Vdot0, F_ext):
    nBatch, nJoint = jointPos.shape
    if jointVel.shape != jointPos.shape or jointAcc.shape != jointPos.shape:
        print(f'[Error] solveRecursiveDynamics: jointVel.shape = {jointVel.shape}, jointAcc.shape = {jointAcc.shape}')
        exit(1)
    if Vdot0.shape != (6,) or A_screw.shape != (nJoint, 6) or initialLinkFrames.shape != (nJoint, 4, 4):
        print(f'[Error] solveRecursiveDynamics: Vdot0.shape = {Vdot0.shape}, A_screw.shape = {A_screw.shape}, initialLinkFrames.shape = {initialLinkFrames.shape}')
        exit(1)
    if Phi.shape != (10 * nJoint,) or F_ext.shape != (6,):
        print(f'[Error] solveRecursiveDynamics: Phi.shape = {Phi.shape}, F_ext.shape = {F_ext.shape}')
        exit(1)
    # Forward
    V, Vdot, linkFrames = solveKinematics(jointPos, jointVel, jointAcc, A_screw, initialLinkFrames, Vdot0)
    # Backward
    G = PhiToG(Phi)  # Link inertia, (nJoint, 6, 6)
    F = jointPos.new_zeros(nBatch, nJoint, 6)
    jointTorque = jointPos.new_zeros(nBatch, nJoint)
    F_child = F_ext.repeat(nBatch, 1)  # (nBatch, 6)
    W = jointPos.new_zeros(nBatch, 6 * nJoint, 10 * nJoint)
    Y = jointPos.new_zeros(nBatch, nJoint, 10 * nJoint)
    for i in reversed(range(nJoint)):
        A_i = A_screw[i, :].repeat(nBatch, 1)  # (nBatch, 6)
        G_i = G[i, :, :]  # (6, 6)
        V_i = V[:, i, :]  # (nBatch, 6)
        Vdot_i = Vdot[:, i, :]  # (nBatch, 6)
        pos = jointPos[:, i].view(nBatch, 1)
        vel = jointVel[:, i].view(nBatch, 1)
        acc = jointAcc[:, i].view(nBatch, 1)
        if i < nJoint - 1:
            A_ip1 = A_screw[i + 1, :].repeat(nBatch, 1)  # (nBatch, 6)
            T_ip1 = linkFrames[:, i + 1, :, :]  # (nBatch, 4, 4)
        else:
            A_ip1 = jointPos.new_zeros(nBatch, 6)
            T_ip1 = torch.eye(4).to(jointPos).repeat(nBatch, 1, 1)  # (nBatch, 4, 4)
        AdjInvT_ip1 = largeAdjoint(invSE3(T_ip1))  # (nBatch, 6, 6)
        adV_i = smallAdjoint(V_i)  # (nBatch, 6, 6)
        # recursion
        W_diagonal = V_regressor(Vdot_i) - adV_i.transpose(1, 2) @ V_regressor(V_i)  # (nBatch, 6, 10)
        if i == nJoint - 1:
            W[:, 6 * i:6 * (i + 1), 10 * i:] = W_diagonal
        else:
            W_offdiag = AdjInvT_ip1.transpose(1, 2) @ W[:, 6 * (i + 1):6 * (i + 2), 10 * (i + 1):]  # (nBatch, 6, -1)
            W[:, 6 * i:6 * (i + 1), 10 * i:] = torch.cat([W_diagonal, W_offdiag], dim=2)
        Y[:, i, :] = (A_i.reshape(nBatch, 1, 6) @ W[:, 6 * i:6 * (i + 1), :]).squeeze()  # (nBatch, 10 * nJoint)
        # F_i (nBatch, nJoint, 6)
        F_i = (F_child.reshape(nBatch, 1, 6) @ AdjInvT_ip1).squeeze() + (G_i @ Vdot_i.reshape(nBatch, 6, 1)).squeeze() \
              - (adV_i.transpose(1, 2) @ (G_i @ V_i.reshape(nBatch, 6, 1))).squeeze()  # (nBatch, 6)
        tau_i = (A_i.reshape(nBatch, 1, 6) @ F_i.view(nBatch, 6, 1)).squeeze()  # (nBatch)
        # log
        F_child = F_i
        F[:, i, :] = F_i
        jointTorque[:, i] = tau_i
    return V, Vdot, linkFrames, jointTorque, F, W, Y


def PhiToG(Phi):
    # m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz
    nJoint = int(Phi.numel() / 10)
    assert Phi.shape == (10 * nJoint,), f'[Error] PhiToG: Phi.shape = {Phi.shape}'
    G = Phi.new_zeros(nJoint, 6, 6)
    inertiaMatrix = Phi.reshape(nJoint, 10)
    # mass
    G[:, 3:, 3:] = torch.eye(3).to(Phi).repeat(nJoint, 1, 1) * inertiaMatrix[:, 0].view(nJoint, 1, 1)
    # COM
    h_bracket = skew_so3(inertiaMatrix[:, 1:4])
    G[:, :3, 3:] = h_bracket
    G[:, 3:, :3] = h_bracket.transpose(1, 2)
    # Moment of inertia
    I = inertiaMatrix[:, 4:].reshape(nJoint, 6, 1, 1)
    G[:, :3, :3] = torch.cat([torch.cat([I[:, 0], I[:, 3], I[:, 4]], dim=2),
                              torch.cat([I[:, 3], I[:, 1], I[:, 5]], dim=2),
                              torch.cat([I[:, 4], I[:, 5], I[:, 2]], dim=2)], dim=1)
    return G


def V_regressor(V):
    nBatch, mustbe6 = V.shape  # (nBatch, 6)
    assert mustbe6 == 6, f'[Error] V_regressor: V.shape = {V.shape}'
    V_reg = V.new_zeros(nBatch, 6, 10)
    w = V[:, :3]  # (nBatch, 3)
    v = V[:, 3:]  # (nBatch, 3)
    zeroBatch = V.new_zeros(nBatch, 1, 1)
    V_reg[:, 3:, 0] = v
    V_reg[:, :3, 1:4] = -skew_so3(v)
    V_reg[:, 3:, 1:4] = skew_so3(w)
    V_reg[:, :3, 4:7] = torch.diag_embed(w, dim1=1, dim2=2)
    w_rsh = w.reshape(nBatch, 3, 1, 1)
    V_reg[:, :3, 7:] = torch.cat([torch.cat([w_rsh[:, 1], w_rsh[:, 2], zeroBatch], dim=2),
                                  torch.cat([w_rsh[:, 0], zeroBatch, w_rsh[:, 2]], dim=2),
                                  torch.cat([zeroBatch, w_rsh[:, 0], w_rsh[:, 1]], dim=2)], dim=1)
    return V_reg
