from torch import jit
from day01.net import *
import torch

if __name__ == '__main__':
    input = torch.rand(1,784)
    model = NetV1()
    model.load_state_dict(torch.load("./checkpoint/3.pkl"))
    traced_script_modele = torch.jit.trace(model,input)
    traced_script_modele.save("mnist.pt")