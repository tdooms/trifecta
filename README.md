## Official Implementation of The Trifecta Algorithm

This is the official Pytorch (lightning) code implementation of TFF, The Trifecta Forward-Forward algorithm. 


**Goal.** The Trifecta aims to solve three weaknesses of the Forward-Forward algorithm (Hinton 2022). 
This allows the training of deeper networks which can achieve higher accuracy. Each observed weakness is addressed through a simple modification of the Forward-Forward algorithm.

- **High reliance on the threshold parameter:** We substitute the threshold-based loss function to a distance-based loss function. This allows the network to learn the separation instead of hard-coding it.
- **Limited progressive Improvement:** We introduce BatchNorm into the network, which retains prediction information while forcing the network to lean new features.
- **Lack of error signals:** Instead of learning each layer locally, we perform a semi-local approach that updates two layers at a time. These partial error signals allow each layer to improve its subsequent layers which improves accuracy.

**Code.** The code, written in Pytorch, is organised to be both flexible and readable. 
The full implementation is only 300 lines long, including boilerplate. 
It is written in PyTorch Lightning, which automates logging and improves code readability.

Any questions or comments can be sent to <thomas@dooms.eu>.

**Our Paper.** <https://arxiv.org/abs/2311.18130>

**FF Paper.** <https://arxiv.org/abs/2212.13345>
