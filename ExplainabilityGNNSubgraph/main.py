import torch

if __name__ == "__main__":
    test = torch.zeros((2, 3, 4))
    test2 = 2 * torch.ones((2, 3, 4))
    test2 = test2.sum(dim=-1, keepdim=True)
    test[0, 0, 0] = 1
    test[1, 0, 0] = 2
    print(f"test is {test}")
    print(f"test2 is {test2}")

    print(f"test / test2 is {test,test2}")
