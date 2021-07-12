from model.model import *

nJoint, useAdjoint = 3, True
model = AmbidexWristModel(nJoint, useAdjoint)
nBatch = 4
nMotor = 2
motorPos, motorVel, motorAcc = torch.rand(nBatch, nMotor), torch.rand(nBatch, nMotor), torch.rand(nBatch, nMotor)

eps = 1e-6

def test_diffkine():
    motorState = State(motorPos, motorVel, motorAcc)
    jointState = model.getJointStates(motorState)
    jointPos, jointVel, jointAcc = jointState.pos, jointState.vel, jointState.acc

    assert jointPos.shape == (nBatch, nJoint), f'jointPos.shape = {jointPos.shape}'
    assert jointVel.shape == (nBatch, nJoint), f'jointVel.shape = {jointVel.shape}'
    assert jointAcc.shape == (nBatch, nJoint), f'jointAcc.shape = {jointAcc.shape}'
    jac = model.getJacobian(motorPos)
    assert jac.shape == (nBatch, nJoint, nMotor), f'jac.shape = {jac.shape}'
    assert torch.all((jac @ motorVel.view(nBatch, nMotor, 1)).squeeze() - jointVel < eps)


    # Check gradient
    # motorPos.requires_grad_()
    # assert torch.autograd.gradcheck(func=model.getJointAngle, inputs=(motorPos,))
    input = torch.randn(20, 2, requires_grad=True)
    assert model.getJointAngle(input).shape == (20, nJoint)
    # assert torch.autograd.gradcheck(model.getJointAngle, (input,), eps=1e-6, atol=1e-4,rtol=0.01)

