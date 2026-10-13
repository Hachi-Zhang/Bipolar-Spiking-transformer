import torch
from torch.utils.data import TensorDataset
from seq_cifar.data import SeqCIFAR


def test_sequence_shape_and_row_major_order():
    image=torch.arange(3*32*32,dtype=torch.float32).reshape(1,3,32,32)
    dataset=SeqCIFAR(TensorDataset(image,torch.tensor([0])))
    sequence,label=dataset[0]
    assert sequence.shape==(3,1024) and label.item()==0
    for channel in range(3):
        assert torch.equal(sequence[channel],image[0,channel].reshape(-1))
